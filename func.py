# phenolopy
'''
This script contains functions for calculating per-pixel phenology metrics (phenometrics)
on a timeseries of vegetation index values (e.g. NDVI) stored in a xarray DataArray. The
methodology is based on the TIMESAT 3.3 software. Some elements of Phenolopy were also
inspired by the great work of Chad Burton (chad.burton@ga.gov.au).

The script contains the following primary functions:
        1. removal_outliers: detects and removes timeseries outliers;
        2. interpolate: fill in missing nan values;
        3. smooth: applies one of various smoothers to raw vegetation data;
        4. detect_seasons: count num of seasons (i.e. peaks) in timeseries;
        5. calc_phenometrics: generate phenological metrics on timeseries.
'''

# import required libraries
import os, sys
import xarray as xr
import numpy as np
import pandas as pd
import math
import dask
import rasterio
import warnings
from datetime import datetime, timedelta
from scipy.stats import zscore
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.signal import savgol_filter, find_peaks
from scipy.ndimage import gaussian_filter
from datacube.utils.geometry import assign_crs
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

def remove_outliers(ds, user_factor=2):

    """
    Takes an xarray dataset containing vegetation index variable and removes outliers within
    the timeseries on a per-pixel basis. The resulting dataset contains the timeseries
    with outliers set to nan. Can work on datasets with or without existing nan values.

    Parameters
    ----------
    ds: xarray Dataset
        A two-dimensional or multi-dimensional array containing a vegetation
        index variable (i.e. 'veg_index').
    method: str
        The outlier detection method to apply to the dataset. The median method detects
        outliers by calculating if values in pixel timeseries deviate more than a maximum
        deviation (cutoff) from the median in a moving window (half window width = number
        of values per year / 7) and it is lower than the mean value of its immediate neighbors
        minus the cutoff or it is larger than the highest value of its immediate neighbor plus
        The cutoff is the standard deviation of the entire time-series times a factor given by
        the user.
    user_factor: float
        An value between 0 to 10 which is used to 'multiply' the threshold cutoff. A higher factor
        value results in few outliers (i.e. only the biggest outliers). Default factor is 2.

    Returns
    -------
    ds : xarray Dataset
        The original xarray Dataset inputted into the function, with a all detected outliers in the
        veg_index variable set to nan.
    """
    #print('\nBeginning to remove outliers.')
    # 计算每个像素的标准差，并乘以用户因子
    cutoffs = ds.std('time') * user_factor

    # 计算原始数据中存在的NaN值的掩模
    ds_mask = xr.where(ds.isnull(), True, False)

    # 计算窗口大小
    win_size = max(3, int(len(ds['time']) / 7))
    if win_size % 2 == 0:
        win_size += 1

    # 计算整个数据集的滚动中位数
    ds_med = ds.rolling(time=win_size, center=True).median()

    # 计算滚动中位数的NaN掩模，并替换起始/结束的NaN值
    med_mask = xr.where(ds_med.isnull(), True, False)
    med_mask = xr.where(ds_mask != med_mask, True, False)
    ds_med = xr.where(med_mask, ds, ds_med)

    # 计算原始数据与中位数之间的绝对差异
    ds_diffs = abs(ds - ds_med)

    # 计算异常值掩模
    outlier_mask = xr.where(ds_diffs > cutoffs, True, False)

    # 用0 / nan 替换异常值
    ds_cleaned = xr.where(outlier_mask, float('nan'), ds)

    # Step to set EVI2 values less than 0.08 to 0
    ds_cleaned = xr.where(ds_cleaned < 800, float('nan'), ds_cleaned)

    # 通知用户
    #print('> Success!')
    return ds_cleaned


def interpolate_spline(data, time_values, s=1.0):
    """
    Perform smoothing spline interpolation on all values in the dataset.

    Parameters
    ----------
    data : numpy array
        The input EVI2 values with potential missing values (NaNs).
    time_values : numpy array
        Time values in numeric format (e.g., days since the first observation).
    s : float, optional
        Smoothing factor for the spline. The larger the value, the smoother the spline.
        Default is 1.0.
    
    Returns
    -------
    interpolated_data : numpy array
        Interpolated data over the full time range.
    """
    #print('\nBeginning to interpolate the EVI2 data.')
    # Initialize output array with NaNs
    interpolated_data = np.full_like(data, np.nan)

    # Define a helper function for interpolation on a single time series
    def interpolate_single_pixel(pixel_time_series):
        valid_indices = np.isfinite(pixel_time_series)
        if np.sum(valid_indices) > 3:  # At least 4 valid points required for spline interpolation
            try:
                # Perform smoothing spline interpolation
                spline = UnivariateSpline(time_values[valid_indices], pixel_time_series[valid_indices],k=3,s=s)
                return spline(time_values)
            except Exception:
                # Fallback to linear interpolation if spline fails
                return np.interp(time_values, time_values[valid_indices], pixel_time_series[valid_indices])
        else:
            # If insufficient valid data points, return original values
            return pixel_time_series

    # Apply the helper function along each (y, x) pixel's time series
    interpolated_data = np.apply_along_axis(interpolate_single_pixel, 0, data)

    #print('> Interpolation successful.')
    return interpolated_data



def smooth_sg(ds, window_length=91, polyorder=1):
    """
    Takes an xarray dataset containing vegetation index variable and smoothes timeseries
    timeseries on a per-pixel basis. The resulting dataset contains a smoother timeseries.
    Recommended that no nan values present in dataset.

    Parameters
    ----------
    ds: xarray Dataset
        A two-dimensional or multi-dimensional array containing a vegetation
        index variable (i.e. 'veg_index').
    window_length: int
        The length of the filter window (i.e., the number of coefficients). Value must
        be a positive odd integer. The larger the window length, the smoother the dataset.
        Default value is 3 (as per TIMESAT).
    polyorder: int
        The order of the polynomial used to fit the samples. Must be a odd number (int) and
        less than window_length.

    Returns
    -------
    ds : xarray Dataset
        The original xarray Dataset as input into the function, with smoothed data in the
        veg_index variable.
    """

    # notify user
    #print('\nSmoothing method: Savitzky-Golay with window length: {0} and polyorder: {1}.'.format(window_length, polyorder))

    # check if type is xr dataset
    if type(ds) != xr.Dataset:
        raise TypeError('> Not a dataset. Please provide a xarray dataset.')

    # check if time dimension is in dataset
    if 'time' not in list(ds.dims):
        raise ValueError('> Time dimension not in dataset. Please ensure dataset has a time dimension.')

    # check if window length provided
    if window_length <= 0 or not isinstance(window_length, int):
        raise TypeError('> Window_length is <= 0 and/or not an integer. Please provide a value of 0 or above.')

    # check if user factor provided
    if polyorder <= 0 or not isinstance(polyorder, int):
        raise TypeError('> Polyorder is <= 0 and/or not an integer. Please provide a value of 0 or above.')

    # check if polyorder less than window_length
    if polyorder > window_length:
        raise TypeError('> Polyorder is > than window_length. Must be less than window_length.')

    # create savitsky smoother func
    def smoother(da, window_length, polyorder):
        return da.apply(savgol_filter, window_length=window_length, polyorder=polyorder, axis=0)

    # create kwargs dict
    kwargs = {'window_length': window_length, 'polyorder': polyorder}

    # create template and map func to dask chunks
    temp = xr.full_like(ds, fill_value=np.nan)
    ds = xr.map_blocks(smoother, ds, template=temp, kwargs=kwargs)

    # check if any nans exist in dataset after resample and tell user
    #if bool(ds.isnull().any()):
    #    print('> Warning: dataset contains nan values. You may want to interpolate next.')

    # notify user
    #print('\n> Smoothing successful.')

    return ds

def plot_evi2_time_series(evi2_ds, evi2_ds_target, evi2_ds_outliers, evi2_ds_interpolated, 
                         y_index, x_index, target_year):
    """
    绘制指定采样点的 EVI2 时间序列，并进行相关统计分析和标记。

    参数:
    - evi2_ds: 原始数据集 (xarray Dataset)
    - evi2_ds_target: 目标平滑数据集 (xarray Dataset)
    - evi2_ds_outliers: 剔除离群值后的数据集 (xarray Dataset)
    - evi2_ds_interpolated: 插值后的数据集 (xarray Dataset)
    - y_index: 样本的行索引
    - x_index: 样本的列索引
    - target_year: 要分析的目标年份 
    """
    # 提取目标时间序列数据
    selected_data_smooth_target = evi2_ds_target['EVI2'][:, y_index, x_index].values
    time_values_target = evi2_ds_target['time']
    
    # 提取其他时间序列
    time_series_original = evi2_ds['EVI2'].sel(y=y_index, x=x_index).values
    time_series_outliers = evi2_ds_outliers['EVI2'].sel(y=y_index, x=x_index).values
    time_series_interpolated = evi2_ds_interpolated['EVI2'].sel(y=y_index, x=x_index).values
    time_values = evi2_ds.coords['time'].values

    # 将时间转换为 pandas 日期时间格式
    time_values_dt = pd.to_datetime(time_values_target)

    # 筛选出目标年份的数据
    mask = time_values_dt.year == target_year
    selected_data_target_year = selected_data_smooth_target[mask]
    time_values_target_year = time_values_dt[mask]

    # 如果目标年份的数据为空
    if len(selected_data_target_year) == 0:
        print(f'No data available for the year {target_year}.')
        return

    # 计算目标年份的统计量
    mean_value = np.mean(selected_data_smooth_target)
    max_value = np.max(selected_data_target_year)
    max_index = np.argmax(selected_data_target_year)

    # 计算差分斜率
    slopes1 = np.diff(selected_data_target_year)
    slopes2 = np.diff(selected_data_smooth_target)

    # 查找目标年份一阶导数从正变为零的所有点
    zero_cross_indices = [
        i for i in range(1, len(slopes1)) if slopes1[i-1] > 0 and slopes1[i] <= 0
    ]

    # 查找最大值左侧的最小值
    left_min_index = None
    for i in range(0, 182 + max_index):
        if slopes2[i] >= 0:
            if left_min_index is None or selected_data_smooth_target[i] < selected_data_smooth_target[left_min_index]:
                left_min_index = i

    # 查找最大值右侧的最小值
    right_min_index = None
    for i in range(max_index + 182 + 1, len(selected_data_smooth_target)):
        if slopes2[i - 1] <= 0:
            if right_min_index is None or selected_data_smooth_target[i] < selected_data_smooth_target[right_min_index]:
                right_min_index = i

    # 绘图
    plt.figure(figsize=(20, 6))
    plt.plot(time_values_dt, selected_data_smooth_target, label='Smoothed', linestyle='-', color='red')
    plt.fill_between(time_values_target_year, 
                     np.min(selected_data_smooth_target), 
                     np.max(selected_data_smooth_target), 
                     color='lightblue', alpha=0.2, label='Target Year Area')
    plt.scatter(time_values_target_year[max_index], max_value, color='blue', label='Max Value', zorder=5)
    plt.text(time_values_target_year[max_index], max_value, f'Max: {max_value:.2f}', fontsize=12, color='blue')

    if left_min_index is not None:
        left_min_value = selected_data_smooth_target[left_min_index]
        plt.scatter(time_values_dt[left_min_index], left_min_value, color='green', label='Left Min', zorder=5)
        plt.text(time_values_dt[left_min_index], left_min_value, f'Min: {left_min_value:.2f}', fontsize=12, color='green')

    if right_min_index is not None:
        right_min_value = selected_data_smooth_target[right_min_index]
        plt.scatter(time_values_dt[right_min_index], right_min_value, color='red', label='Right Min', zorder=5)
        plt.text(time_values_dt[right_min_index], right_min_value, f'Min: {right_min_value:.2f}', fontsize=12, color='red')

    for zero_cross_index in zero_cross_indices:
        zero_cross_time = time_values_target_year[zero_cross_index]
        zero_cross_value = selected_data_target_year[zero_cross_index]
        plt.scatter(zero_cross_time, zero_cross_value, color='purple', label='Zero Crossing', zorder=5)
        plt.text(zero_cross_time, zero_cross_value, f'Zero: {zero_cross_value:.2f}', fontsize=12, color='purple')

    start_of_year = pd.Timestamp(f'{target_year}-01-01')
    end_of_year = pd.Timestamp(f'{target_year}-12-31')
    #plt.xlim(start_of_year, end_of_year)
    plt.ylim(-1,)
    plt.axhline(mean_value, color='orange', linestyle='--', label=f'Mean: {mean_value:.2f}')
    plt.plot(time_values, time_series_original, label='Original', marker='', color='blue', alpha=0.6)
    plt.plot(time_values, time_series_outliers, label='Outliers Removed', marker='', color='orange', alpha=0.6)
    plt.plot(time_values, time_series_interpolated, label='Interpolated', marker='', color='green', alpha=0.6)
    plt.title(f'EVI2 Time Series at row={y_index}, col={x_index} ({target_year})', fontsize=16)
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gcf().autofmt_xdate()
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('EVI2 Value', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.show()

    # 打印结果
    print(f'row={y_index}, col={x_index} ({target_year}):')
    print(f'Mean EVI2: {mean_value:.2f}')
    print(f'Max EVI2: {max_value:.2f} at time {time_values_target_year[max_index].strftime("%Y-%m-%d")}')
    if left_min_index is not None:
        print(f'Left Min: {left_min_value:.2f} at time {time_values_dt[left_min_index].strftime("%Y-%m-%d")}')
    if right_min_index is not None:
        print(f'Right Min: {right_min_value:.2f} at time {time_values_dt[right_min_index].strftime("%Y-%m-%d")}')
    for zero_cross_index in zero_cross_indices:
        zero_cross_time = time_values_target_year[zero_cross_index].strftime('%Y-%m-%d')
        zero_cross_value = selected_data_target_year[zero_cross_index]
        print(f'Zero Cross: {zero_cross_value:.2f} at time {zero_cross_time}')

def extract_crs(da):
    """
    Takes an xarray Dataset pulled from opendatacube and extracts crs metadata
    if exists. Returns None if not found.

    Parameters
    ----------
    da: xarray Dataset
        A single- or multi-dimensional array containing (or not) crs metadata.

    Returns
    -------
    crs: str
        A crs object.
    """

    # notify user
    print('Beginning extraction of CRS metadata.')
    try:
        # notify user
        print('> Extracting CRS metadata.')
        # extract crs metadata
        crs = da.geobox.crs
        # notify user
        print('> Success!\n')
    except:
        # notify user
        print('> No CRS metadata found. Returning None.\n')
        crs = None
    return crs

def add_crs(ds, crs):
    """
    Takes an xarray Dataset adds previously extracted crs metadata, if exists.
    Returns None if not found.

    Parameters
    ----------
    ds: xarray Dataset
        A single- or multi-dimensional array with/without crs metadata.

    Returns
    -------
    ds: xarray Dataset
        A Dataset with a new crs.
    """

    # notify user
    print('Beginning addition of CRS metadata.')
    try:
        # notify user
        print('> Adding CRS metadata.')
        # assign crs via odc utils
        ds = assign_crs(ds, str(crs))
        # notify user
        print('> Success!\n')

    except:
        # notify user
        print('> Could not add CRS metadata to data. Aborting.\n')
        pass
    return ds


def get_mean(da):
    """
    计算 xarray DataArray 的均值。
    """
    # notify user
    #print('Beginning to calculate of mean value of season across total time series.')   
    
    da_mean_values = da.mean(dim='time') 
    da_mean_values = xr.DataArray(da_mean_values.astype('float32'), dims=('y', 'x'))
    da_mean_values = da_mean_values.rename('mean_values')
    
    # notify user
    #print('> Success!\n')
      
    return da_mean_values



def DOY(time, target_year):
    """
    计算给定时间相对于target_year-01-01的天数。

    参数：
    ----------
    time : pd.Timestamp, datetime
        要计算的时间。
    target_year : int, 
        参考的目标年份。

    返回：
    -------
    相对target_year 1月1日的天数，如果是target_year的日期返回dayofyear。
    """
    # 将 numpy.datetime64 或其他 datetime-like 转换为 pandas.Timestamp
    time = pd.to_datetime(time)

    # 创建 target_year 的 1月1日的时间戳
    target_start = pd.Timestamp(f'{target_year}-01-01')

    # 计算 time 和 target_start 的时间差
    delta = (time - target_start).days

    # 如果 time 在 target_year 年份内，返回 dayofyear
    if time.year == target_year:
        return time.dayofyear
    # 否则，返回相对于 target_year 1月1日的天数差
    else:
        return delta


def calculate_peak_and_season(y, x, data, mean_values, mask, time_dt, target_year):
    """
    Function to calculate the number of peaks (seasons) and peak times/values for each pixel.
    
    Parameters
    ----------
    y : int
        The y coordinate (row index) for the pixel.
    x : int
        The x coordinate (column index) for the pixel.
    data : np.ndarray
        The 3D data array (time, y, x).
    mean_values : np.ndarray
        The 2D array of mean values for each pixel across the time series.
    mask : np.ndarray
        A boolean array indicating whether the time corresponds to the target year.
    time_dt : pd.DatetimeIndex
        The datetime array corresponding to the times.
    target_year : int
        The target year for which to calculate the number of seasons.
        
    Returns
    -------
    tuple
        A tuple containing:
        - Number of seasons (da_nos)
        - First peak time (da_pos_times1)
        - First peak value (da_pos_values1)
        - Second peak time (da_pos_times2)
        - Second peak value (da_pos_values2)
    """
    
    selected_data = data[:, y, x][mask]
    mean_value = mean_values[y, x]

    da_nos = 0
    da_pos_times1 = np.nan
    da_pos_values1 = np.nan
    da_pos_times2 = np.nan
    da_pos_values2 = np.nan

    if len(selected_data) > 0:
        slopes = np.diff(selected_data)
        zero_cross_indices = np.where((slopes[:-1] > 0) & (slopes[1:] <= 0))[0] + 1
        
        candidate_peaks = selected_data[zero_cross_indices]
        candidate_times = time_dt[mask][zero_cross_indices]
        
        valid_peaks = [(p, t) for p, t in zip(candidate_peaks, candidate_times) if p >= mean_value]

        if len(valid_peaks) > 0:
            peak_info = sorted(valid_peaks, key=lambda x: x[0], reverse=True)[:2]
            peak_info_sorted = sorted(peak_info, key=lambda x: x[1])  # Sort by time
            
            if len(peak_info_sorted) == 1:
                peak1_value, peak1_time = peak_info_sorted[0]
                da_nos = 1
                da_pos_values1 = peak1_value
                da_pos_times1 = DOY(peak1_time, target_year)

            elif len(peak_info_sorted) == 2:
                peak1_value, peak1_time = peak_info_sorted[0]
                peak2_value, peak2_time = peak_info_sorted[1]
                da_nos = 2

                time_diff = (peak2_time - peak1_time).days

                if time_diff < 90:
                    if peak1_value > peak2_value:
                        da_nos = 1
                        da_pos_values1 = peak1_value
                        da_pos_times1 = DOY(peak1_time, target_year)
                    else:
                        da_nos = 1
                        da_pos_values1 = peak2_value
                        da_pos_times1 = DOY(peak2_time, target_year)
                else:
                    da_pos_values1 = peak1_value
                    da_pos_times1 = DOY(peak1_time, target_year)
                    da_pos_values2 = peak2_value
                    da_pos_times2 = DOY(peak2_time, target_year)

    return da_nos, da_pos_times1, da_pos_values1, da_pos_times2, da_pos_values2


def get_pos(da, mean_values, target_year, n_jobs):
    """
    Optimized version of the original function to calculate the number of seasons 
    and peaks for each pixel in the target year.
    
    Parameters
    ----------
    da : xarray.DataArray
        A 3D array of vegetation index values with time, y, and x dimensions.
    mean_values : xarray.DataArray
        A 2D array of mean values for each pixel across the time series (not for the target year).
    target_year : int
        The target year for which to calculate the number of seasons.
    n_jobs : int, optional
        Number of parallel jobs to run (default is 4).

    Returns
    -------
    da_nos, da_pos_times1, da_pos_values1, da_pos_times2, da_pos_values2 : xarray.DataArray
        Xarray DataArrays containing the number of seasons and peak information for each pixel.
    """
    
    #print('Beginning calculation of peak of season (pos) values and times and number of seasons (nos).')

    # 获取时间和数据
    time = da['time'].values
    time_dt = pd.to_datetime(time)
    mask = (time_dt.year == target_year)
    
    data = da.values  # 数据现在是一个三维数组 (time, y, x)

    da_nos = np.zeros((data.shape[1], data.shape[2]), dtype=int)
    da_pos_times1 = np.full((data.shape[1], data.shape[2]), np.nan)
    da_pos_values1 = np.full((data.shape[1], data.shape[2]), np.nan)
    da_pos_times2 = np.full((data.shape[1], data.shape[2]), np.nan)
    da_pos_values2 = np.full((data.shape[1], data.shape[2]), np.nan)

    results = Parallel(n_jobs=n_jobs)(
        delayed(calculate_peak_and_season)(y, x, data, mean_values, mask, time_dt, target_year)
        for y in range(data.shape[1]) for x in range(data.shape[2])
    )


    for i, (y, x) in enumerate(((y, x) for y in range(data.shape[1]) for x in range(data.shape[2]))):
        da_nos[y, x], da_pos_times1[y, x], da_pos_values1[y, x], da_pos_times2[y, x], da_pos_values2[y, x] = results[i]


    da_nos = xr.DataArray(da_nos.astype('float32'), dims=('y', 'x'))
    da_pos_times1 = xr.DataArray(da_pos_times1.astype('float32'), dims=('y', 'x'))
    da_pos_values1 = xr.DataArray(da_pos_values1.astype('float32'), dims=('y', 'x'))
    da_pos_times2 = xr.DataArray(da_pos_times2.astype('float32'), dims=('y', 'x'))
    da_pos_values2 = xr.DataArray(da_pos_values2.astype('float32'), dims=('y', 'x'))

    # Rename variables
    da_nos = da_nos.rename('num_seasons')
    da_pos_times1 = da_pos_times1.rename('pos_times1')
    da_pos_values1 = da_pos_values1.rename('pos_values1')
    da_pos_times2 = da_pos_times2.rename('pos_times2')
    da_pos_values2 = da_pos_values2.rename('pos_values2')

    #print('> Success!\n')

    return da_nos, da_pos_times1, da_pos_values1, da_pos_times2, da_pos_values2


def calculate_slope(series):
    """
    Helper function to calculate the slope (difference) of a time series.
    """
    return np.diff(series)

def find_valleys_for_pixel(y, x, da, target_year, da_peak_times1, da_peak_times2, da_peak_values1, da_peak_values2):
    """
    Process a single pixel (y, x) to calculate its valley points.
    """
    time_series = da[:, y, x].values
    peak_time1 = da_peak_times1[y, x].values
    peak_time2 = da_peak_times2[y, x].values if not np.isnan(da_peak_times2[y, x].values) else None

    peak_indices1 = np.where(da['time'].values == peak_time1)[0]
    if len(peak_indices1) == 0:
        return (y, x, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

    peak_index1 = peak_indices1[0]

    left_values1 = time_series[:peak_index1]
    min_left_value1 = np.nan
    min_left_time1 = np.nan
    if len(left_values1) > 0 and not np.all(np.isnan(left_values1)):
        left_slopes1 = calculate_slope(left_values1)
        left_turning_points1 = np.where((left_slopes1[:-1] < 0) & (left_slopes1[1:] >= 0))[0]

        if len(left_turning_points1) > 0:
            min_left_index1 = left_turning_points1[np.nanargmin(left_values1[left_turning_points1])]
            min_left_value1 = left_values1[min_left_index1 + 1]
            min_left_time1 = da['time'].values[min_left_index1 + 1]

    right_values1 = time_series[peak_index1 + 1:]
    min_right_value1 = np.nan
    min_right_time1 = np.nan
    if len(right_values1) > 0 and not np.all(np.isnan(right_values1)):
        right_slopes1 = calculate_slope(right_values1)
        right_turning_points1 = np.where((right_slopes1[:-1] < 0) & (right_slopes1[1:] >= 0))[0]

        if len(right_turning_points1) > 0:
            min_right_index1 = right_turning_points1[np.nanargmin(right_values1[right_turning_points1])]
            min_right_value1 = right_values1[min_right_index1 + 1]
            min_right_time1 = da['time'].values[min_right_index1 + (peak_index1 + 1) + 1]

    min_left_value2 = np.nan
    min_left_time2 = np.nan
    min_right_value2 = np.nan
    min_right_time2 = np.nan

    if peak_time2 is not None:
        peak_indices2 = np.where(da['time'].values == peak_time2)[0]
        if len(peak_indices2) > 0:
            peak_index2 = peak_indices2[0]

            left_values2 = time_series[peak_index1 + 1:peak_index2]
            if len(left_values2) > 0 and not np.all(np.isnan(left_values2)):
                left_slopes2 = calculate_slope(left_values2)
                left_turning_points2 = np.where((left_slopes2[:-1] < 0) & (left_slopes2[1:] >= 0))[0]

                if len(left_turning_points2) > 0:
                    min_left_index2 = left_turning_points2[np.nanargmin(left_values2[left_turning_points2])]
                    min_left_value2 = left_values2[min_left_index2 + 1]
                    min_left_time2 = da['time'].values[min_left_index2 + (peak_index1 + 1) + 1]

            right_values2 = time_series[peak_index2 + 1:]
            if len(right_values2) > 0 and not np.all(np.isnan(right_values2)):
                right_slopes2 = calculate_slope(right_values2)
                right_turning_points2 = np.where((right_slopes2[:-1] < 0) & (right_slopes2[1:] >= 0))[0]

                if len(right_turning_points2) > 0:
                    min_right_index2 = right_turning_points2[np.nanargmin(right_values2[right_turning_points2])]
                    min_right_value2 = right_values2[min_right_index2 + 1]
                    min_right_time2 = da['time'].values[min_right_index2 + (peak_index2 + 1)]

    return (y, x, min_left_time1, min_left_value1, min_right_time1, min_right_value1,
            min_left_time2, min_left_value2, min_right_time2, min_right_value2)

def get_vos(da, target_year, da_peak_times1, da_peak_times2, da_peak_values1, da_peak_values2, n_jobs):
    """
    Optimized function to find the valley points (vos) in the time series data based on the peaks.
    """
    #print('Beginning calculate of valley points (vos) values and times.')

    da_doy = xr.apply_ufunc(lambda time: DOY(time, target_year), da['time'], vectorize=True)
    da = da.assign_coords(time=da_doy)

    shape = da_peak_values1.shape
    da_vos_l_times1 = np.full(shape, np.nan, dtype='float32')
    da_vos_l_values1 = np.full(shape, np.nan, dtype='float32')
    da_vos_r_times1 = np.full(shape, np.nan, dtype='float32')
    da_vos_r_values1 = np.full(shape, np.nan, dtype='float32')
    da_vos_l_times2 = np.full(shape, np.nan, dtype='float32')
    da_vos_l_values2 = np.full(shape, np.nan, dtype='float32')
    da_vos_r_times2 = np.full(shape, np.nan, dtype='float32')
    da_vos_r_values2 = np.full(shape, np.nan, dtype='float32')

    yx_indices = [(y, x) for y in range(shape[0]) for x in range(shape[1])]

    results = Parallel(n_jobs=n_jobs)(delayed(find_valleys_for_pixel)(y, x, da, target_year, da_peak_times1, da_peak_times2, da_peak_values1, da_peak_values2)
                                       for y, x in yx_indices)

    for (y, x, l_time1, l_value1, r_time1, r_value1, l_time2, l_value2, r_time2, r_value2) in results:
        da_vos_l_times1[y, x] = l_time1
        da_vos_l_values1[y, x] = l_value1
        da_vos_r_times1[y, x] = r_time1
        da_vos_r_values1[y, x] = r_value1
        da_vos_l_times2[y, x] = l_time2
        da_vos_l_values2[y, x] = l_value2
        da_vos_r_times2[y, x] = r_time2
        da_vos_r_values2[y, x] = r_value2

    da_vos_l_times1 = xr.DataArray(da_vos_l_times1, dims=('y', 'x'), name='vos_l_times1')
    da_vos_l_values1 = xr.DataArray(da_vos_l_values1, dims=('y', 'x'), name='vos_l_values1')
    da_vos_r_times1 = xr.DataArray(da_vos_r_times1, dims=('y', 'x'), name='vos_r_times1')
    da_vos_r_values1 = xr.DataArray(da_vos_r_values1, dims=('y', 'x'), name='vos_r_values1')
    da_vos_l_times2 = xr.DataArray(da_vos_l_times2, dims=('y', 'x'), name='vos_l_times2')
    da_vos_l_values2 = xr.DataArray(da_vos_l_values2, dims=('y', 'x'), name='vos_l_values2')
    da_vos_r_times2 = xr.DataArray(da_vos_r_times2, dims=('y', 'x'), name='vos_r_times2')
    da_vos_r_values2 = xr.DataArray(da_vos_r_values2, dims=('y', 'x'), name='vos_r_values2')

    #print('> Success!\n')

    return (da_vos_l_times1, da_vos_l_values1, da_vos_r_times1, da_vos_r_values1,
            da_vos_l_times2, da_vos_l_values2, da_vos_r_times2, da_vos_r_values2)

def get_valid_cycle(da_pos_times1, da_pos_values1, da_vos_l_times1, da_vos_l_values1, da_vos_r_times1, da_vos_r_values1,
                    da_pos_times2, da_pos_values2, da_vos_l_times2, da_vos_l_values2, da_vos_r_times2, da_vos_r_values2, 
                    da_nos):
    #print('Beginning update the pos and vos for valid season.')

    da_nos.loc[:, :] = 0
    
    # 存在第二个峰值，但是第一个峰值右侧低谷和第二个峰值的左侧低谷不存在
    # 只保留
    if da_pos_times2 is not None:
        peak1_exists = ~np.isnan(da_pos_times1)
        peak2_exists = ~np.isnan(da_pos_times2)
        vos_r1_missing = np.isnan(da_vos_r_times1) | (da_vos_r_times1 == 0)
        vos_l2_missing = np.isnan(da_vos_l_times2) | (da_vos_l_times2 == 0)
        
        condition_mask = peak1_exists & peak2_exists & vos_r1_missing & vos_l2_missing
        
        da_vos_r_times1.values[condition_mask] = np.where(da_pos_values1.values[condition_mask] > da_pos_values2.values[condition_mask],
                                                          da_vos_r_times2.values[condition_mask], da_vos_r_times1.values[condition_mask])
        da_vos_r_values1.values[condition_mask] = np.where(da_pos_values1.values[condition_mask] > da_pos_values2.values[condition_mask],
                                                           da_vos_r_values2.values[condition_mask], da_vos_r_values1.values[condition_mask])

        mask_peak1_greater = da_pos_values1.values[condition_mask] > da_pos_values2.values[condition_mask]
        da_pos_times1.values[condition_mask] = np.where(mask_peak1_greater, da_pos_times1.values[condition_mask], da_pos_times2.values[condition_mask])
        da_pos_values1.values[condition_mask] = np.where(mask_peak1_greater, da_pos_values1.values[condition_mask], da_pos_values2.values[condition_mask])
        
        da_pos_times2.values[condition_mask] = np.nan
        da_pos_values2.values[condition_mask] = np.nan
        da_vos_l_times2.values[condition_mask] = np.nan
        da_vos_l_values2.values[condition_mask] = np.nan
        da_vos_r_times2.values[condition_mask] = np.nan
        da_vos_r_values2.values[condition_mask] = np.nan
        
        da_nos.values[condition_mask] = 1
        
        #print("Updated peak1 and cleared peak2 where conditions met.")

    # 处理后第一个峰值是否同时存在左右低谷值
    valid_mask1 = (~np.isnan(da_vos_l_times1)|(da_vos_l_times1 == 0)) & (~np.isnan(da_vos_r_times1)|(da_vos_r_times1 == 0)) & ~np.isnan(da_pos_times1)
    da_nos.values[valid_mask1] = 1  
    invalid_mask1 = ~valid_mask1
    da_pos_times1.values[invalid_mask1] = np.nan
    da_pos_values1.values[invalid_mask1] = np.nan
    da_vos_l_times1.values[invalid_mask1] = np.nan
    da_vos_l_values1.values[invalid_mask1] = np.nan
    da_vos_r_times1.values[invalid_mask1] = np.nan
    da_vos_r_values1.values[invalid_mask1] = np.nan

    # 处理后第二个峰值是否同时存在左右低谷值
    if da_pos_times2 is not None:
        valid_mask2 = (~np.isnan(da_vos_l_times2)|(da_vos_l_times2 == 0)) & (~np.isnan(da_vos_r_times2)|(da_vos_r_times2 == 0)) & ~np.isnan(da_pos_times2)
        da_nos.values[valid_mask2] = 2  

        invalid_mask2 = ~valid_mask2
        da_pos_times2.values[invalid_mask2] = np.nan
        da_pos_values2.values[invalid_mask2] = np.nan
        da_vos_l_times2.values[invalid_mask2] = np.nan
        da_vos_l_values2.values[invalid_mask2] = np.nan
        da_vos_r_times2.values[invalid_mask2] = np.nan
        da_vos_r_values2.values[invalid_mask2] = np.nan
    
    # 如果peak1不存在而peak2存在，用peak2替换peak1
    for i, j in np.ndindex(valid_mask1.shape):
        if not valid_mask1[i, j] and valid_mask2[i, j]:
            da_pos_times1.values[i, j] = da_pos_times2.values[i, j]
            da_pos_values1.values[i, j] = da_pos_values2.values[i, j]
            da_vos_l_times1.values[i, j] = da_vos_l_times2.values[i, j]
            da_vos_l_values1.values[i, j] = da_vos_l_values2.values[i, j]
            da_vos_r_times1.values[i, j] = da_vos_r_times2.values[i, j]
            da_vos_r_values1.values[i, j] = da_vos_r_values2.values[i, j]
            
            da_pos_times2.values[i, j] = np.nan
            da_pos_values2.values[i, j] = np.nan
            da_vos_l_times2.values[i, j] = np.nan
            da_vos_l_values2.values[i, j] = np.nan
            da_vos_r_times2.values[i, j] = np.nan
            da_vos_r_values2.values[i, j] = np.nan
            
            da_nos.values[i, j] = 1  
    
    if not valid_mask1.any() and (da_pos_times2 is None or not valid_mask2.any()):
        da_nos.loc[:, :] = 0
        
    # 如果 da_pos_times1 是 NaN，则相应的 da_pos_values1, da_vos_l_values1, da_vos_r_values1 也为 NaN
    pos_times1_nan_mask = np.isnan(da_pos_times1)
    da_pos_values1.values[pos_times1_nan_mask] = np.nan
    da_vos_l_values1.values[pos_times1_nan_mask] = np.nan
    da_vos_r_values1.values[pos_times1_nan_mask] = np.nan

    # 如果 da_pos_times2 是 NaN，则相应的 da_pos_values2, da_vos_l_values2, da_vos_r_values2 也为 NaN
    if da_pos_times2 is not None:
        pos_times2_nan_mask = np.isnan(da_pos_times2)
        da_pos_values2.values[pos_times2_nan_mask] = np.nan
        da_vos_l_values2.values[pos_times2_nan_mask] = np.nan
        da_vos_r_values2.values[pos_times2_nan_mask] = np.nan

    #print('> Success!\n')

    return (da_pos_times1, da_pos_values1, da_vos_l_times1, da_vos_l_values1, da_vos_r_times1, da_vos_r_values1,
            da_pos_times2, da_pos_values2, da_vos_l_times2, da_vos_l_values2, da_vos_r_times2, da_vos_r_values2, 
            da_nos)



def get_bse(da_valley_l_values1, da_valley_r_values1,da_valley_l_values2, da_valley_r_values2):
    """
    Calculates the base line (bse) values as the mean of left and right minimum values for each pixel.
    
    Parameters
    ----------
    da_valley_l_values1, da_valley_r_values1 : xarray DataArray
        Left and right valley values for the first cycle.
    da_valley_l_values2, da_valley_r_values2 : xarray DataArray, optional
        Left and right valley values for the second cycle, if applicable.
    
    Returns
    -------
    da_bse_values1 : xarray DataArray
        The baseline for the first cycle, computed as the mean of left and right valley values.
    da_bse_values2 : xarray DataArray or None
        The baseline for the second cycle, computed as the mean of left and right valley values,
        if the second cycle values are provided. Otherwise, None is returned.
    """
    
    # Notify user
    #print('Beginning to calculate the base line values.')

    # Calculate base line for the first cycle
    da_bse_values1 = (da_valley_l_values1 + da_valley_r_values1) / 2
    
    # Initialize the second base line as None
    da_bse_values2 = None
    
    # If second cycle values are provided, calculate the base line for the second cycle
    if da_valley_l_values2 is not None and da_valley_r_values2 is not None:
        da_bse_values2 = (da_valley_l_values2 + da_valley_r_values2) / 2
    
    # Handle NaNs by using np.nanmean
    da_bse_values1 = da_bse_values1.where(~np.isnan(da_bse_values1), np.nan)
    da_bse_values2 = da_bse_values2.where(~np.isnan(da_bse_values2), np.nan)
    
    # Convert type if necessary
    da_bse_values1 = da_bse_values1.astype('float32').rename('bse_values1')
    da_bse_values2 = da_bse_values2.astype('float32').rename('bse_values2')
    
    # Notify user
    #print('> Success!\n')
    
    return da_bse_values1, da_bse_values2


def get_aos(da_peak_values1, da_valley_l_values1, da_valley_r_values1, da_peak_values2, da_valley_l_values2, da_valley_r_values2):
    """
    Calculates the amplitude of season evaluation (AOS) values.

    Parameters
    ----------
    da_peak_values: xarray DataArray
        An xarray DataArray type with the peak values for each pixel.
    da_valley_l_values: xarray DataArray
        An xarray DataArray type with the left minimum values for each pixel.
    da_valley_r_values: xarray DataArray
        An xarray DataArray type with the right minimum values for each pixel.
    da_base_values: xarray DataArray
        An xarray DataArray type containing the mean values of left and right
        minimums for each pixel (BSE).

    Returns
    -------
    da_aos_l_values: xarray DataArray
        An xarray DataArray type containing the differences between peak values
        and left minimum values.
    da_aos_r_values: xarray DataArray
        An xarray DataArray type containing the differences between peak values
        and right minimum values.
    da_aos_values: xarray DataArray
        An xarray DataArray type containing the overall amplitude values.
    """
    # notify user
    #print('Beginning to calculate the amplitude of season (aos) values (times not possible).')

    # Calculate differences for left and right values of the first season
    da_aos_l_values1 = da_peak_values1 - da_valley_l_values1
    da_aos_r_values1 = da_peak_values1 - da_valley_r_values1

    # Calculate overall amplitude
    da_aos_values1 = da_peak_values1 - (da_valley_l_values1 + da_valley_r_values1) / 2

    # Calculate differences for left and right values of the second season
    da_aos_l_values2 = da_peak_values2 - da_valley_l_values2
    da_aos_r_values2 = da_peak_values2 - da_valley_r_values2

    # Calculate overall amplitude
    da_aos_values2 = da_peak_values2 - (da_valley_l_values2 + da_valley_r_values2) / 2

    # Convert types if necessary
    da_aos_l_values1 = da_aos_l_values1.astype('float32').rename('aos_l_values1')
    da_aos_r_values1 = da_aos_r_values1.astype('float32').rename('aos_r_values1')
    da_aos_values1 = da_aos_values1.astype('float32').rename('aos_values1')
    da_aos_l_values2 = da_aos_l_values2.astype('float32').rename('aos_l_values2')
    da_aos_r_values2 = da_aos_r_values2.astype('float32').rename('aos_r_values2')
    da_aos_values2 = da_aos_values2.astype('float32').rename('aos_values2')

    # notify user
    #print('> Success!\n')

    return da_aos_l_values1, da_aos_r_values1, da_aos_values1, da_aos_l_values2, da_aos_r_values2, da_aos_values2


def get_sos(da, target_year, da_valley_l_times1, da_valley_l_values1, da_amplitude_l_values1, da_peak_times1,
            da_valley_l_times2=None, da_valley_l_values2=None, da_amplitude_l_values2=None, da_peak_times2=None,
            factor=0.5):

    """
    Takes several xarray DataArrays containing the highest vege values and times (pos or mos),
    the lowest vege values (bse or vos), and the amplitude (aos) values and calculates the
    vegetation values and times at the start of season (sos). Several methods can be used to
    detect the start of season; most are based on TIMESAT 3.3 methodology.

    Parameters
    ----------
    da : xarray DataArray
        A two-dimensional or multi-dimensional array containing an DataArray of veg_index
        and time values.
    da_valley_l_times: xarray DataArray
        An xarray DataArray with the times of the left minimum values for each pixel.
    da_valley_l_values: xarray DataArray
        An xarray DataArray with the left minimum values for each pixel.
    da_amplitude_l_values : xarray DataArray
        An xarray DataArray type with an x and y dimension (no time). Each pixel is the
        amplitude of season (aos) value detected between the peak and left minimum values
        across the timeseries at each pixel.
    da_peak_times: xarray DataArray
        An xarray DataArray type with an x and y dimension (no time). Each pixel is the
        time (day of year) value detected at either the peak (pos) or middle (mos) of
        season.
    factor: float
        A float value between 0 and 1 which is used to increase or decrease the amplitude
        threshold for the seasonal_amplitude method. A factor closer to 0 results in start
        of season nearer to min value, a factor closer to 1 results in start of season
        closer to peak of season.

    Returns
    -------
    da_sos_values : xarray DataArray
        An xarray DataArray type with an x and y dimension (no time). Each pixel is the
        veg_index value detected at the start of season (sos).
    da_sos_times : xarray DataArray
        An xarray DataArray type with an x and y dimension (no time). Each pixel is the
        time (day of year) value detected at the start of season (sos).
    """

    # notify user
    #print('Beginning to calculate the start of season (sos) values and times.')

    # check factor
    if factor < 0 or factor > 1:
        raise ValueError('Provided factor value is not between 0 and 1. Aborting.')

    # Convert da['time'] to DOY for easier indexing
    da_doy = xr.apply_ufunc(
        lambda time: DOY(time, target_year),
        da['time'],
        vectorize=True
    )
    da = da.assign_coords(time=da_doy).transpose('time', 'y', 'x')
    #print('da.time:',da.time.values)
    #print("da.time.dtype:", da.time.dtype)
    #print(len(set(da.time)))

    # Initialize output arrays
    da_sos_times1 = xr.full_like(da_peak_times1, np.nan, dtype='float32')
    da_sos_values1 = xr.full_like(da_peak_times1, np.nan, dtype='float32')
    da_sos_times2 = xr.full_like(da_peak_times1, np.nan, dtype='float32')
    da_sos_values2 = xr.full_like(da_peak_times1, np.nan, dtype='float32')

    #print('Beginning to calculate the start of the first season.')
    sos_target_value1 = da_valley_l_values1 + da_amplitude_l_values1 * factor

    # Find the SOS1 point in the data array (first peak)
    da_sos_times1 = xr.where((da.time >= da_valley_l_times1) & (da.time <= da_peak_times1) &
                             (da >= sos_target_value1), da.time, np.nan).min(dim='time')
    da_sos_times1 = xr.where((da_sos_times1 >= da.time.min()) & (da_sos_times1 <= da.time.max()),
                             da_sos_times1, np.nan)

    #da_sos_times1 = da_sos_times1.astype(da.time.dtype)
    #print('da_sos_times1:',da_sos_times1.values)
    #print("da_sos_times1.dtype:", da_sos_times1.dtype)

    # Interpolating da to match the identified sos times
    #da_sos_values1 = da.sel(time=da_sos_times1, method='nearest')
    
    sos_indices1 = da.time == da_sos_times1
    da_sos_values1 = da.where(sos_indices1, drop=False).isel(time=0)
    #print('da_sos_values1:',da_sos_values1.values)
    #print('> Success!\n')

    #print('Beginning to calculate the start of the second season.')
    # Initialize sos2 as None for cases without a second peak
    da_sos_times2, da_sos_values2 = None, None

    # Check and calculate SOS for second peak if available
    if da_valley_l_times2 is not None and da_amplitude_l_values2 is not None:
        sos_target_value2 = da_valley_l_values2 + da_amplitude_l_values2 * factor
        da_sos_times2 = xr.where((da.time >= da_valley_l_times2) & (da.time <= da_peak_times2) &
                                 (da >= sos_target_value2), da.time, np.nan).min(dim='time')
        da_sos_times2 = xr.where((da_sos_times2 >= da.time.min()) & (da_sos_times2 <= da.time.max()),
                                da_sos_times2, np.nan)

        # Interpolating da to match the identified sos times for the second peak
        #da_sos_values2 = da.sel(time=da_sos_times2, method='nearest')
        sos_indices2 = da.time == da_sos_times2
        da_sos_values2 = da.where(sos_indices2, drop=False).isel(time=0)

    # 如果 da_sos_times1 是 NaN，则相应的 da_sos_values1 也设为 NaN
    sos_times1_nan_mask = np.isnan(da_sos_times1)
    da_sos_values1 = xr.where(~sos_times1_nan_mask, da_sos_values1, np.nan)
    
    # 如果 da_sos_times2 是 NaN，则相应的 da_sos_values2 也设为 NaN
    if da_sos_times2 is not None:
        sos_times2_nan_mask = np.isnan(da_sos_times2)
        da_sos_values2 = xr.where(~sos_times2_nan_mask, da_sos_values2, np.nan)

    da_sos_times1 = da_sos_times1.astype('float32').rename('sos_times1')
    da_sos_values1 = da_sos_values1.astype('float32').rename('sos_values1')
    da_sos_times2 = da_sos_times2.astype('float32').rename('sos_times2')
    da_sos_values2 = da_sos_values2.astype('float32').rename('sos_values2')

    #print('> Success!\n')

    return da_sos_times1, da_sos_values1, da_sos_times2, da_sos_values2


def get_eos(da, target_year, da_valley_r_times1, da_valley_r_values1, da_amplitude_r_values1, da_peak_times1,
            da_valley_r_times2=None, da_valley_r_values2=None, da_amplitude_r_values2=None, da_peak_times2=None,
            factor=0.5):
    """
    Calculates the vegetation values and times at the end of season (eos).
    The eos is defined as the time when the vegetation index decreases to a
    certain threshold based on the amplitude from the peak value.

    Parameters
    ----------
    da : xarray DataArray
        A two-dimensional or multi-dimensional array containing an DataArray of veg_index
        and time values.
    da_peak_times: xarray DataArray
        An xarray DataArray type with an x and y dimension (no time). Each pixel is the
        time (day of year) value detected at the peak of season.
    da_amplitude_r_values : xarray DataArray
        An xarray DataArray type with an x and y dimension (no time). Each pixel is the
        amplitude of season (aos) value detected between the peak and right minimum values
        across the timeseries at each pixel.
    factor: float
        A float value between 0 and 1 which is used to define the threshold for the eos.

    Returns
    -------
    da_eos_values : xarray DataArray
        An xarray DataArray type with an x and y dimension (no time). Each pixel is the
        veg_index value detected at the end of season (eos).
    da_eos_times : xarray DataArray
        An xarray DataArray type with an x and y dimension (no time). Each pixel is the
        time (day of year) value detected at the end of season (eos).
    """

    # Notify user
    #print('Beginning to calculate the end of season (eos) values and times.')

    # Check factor
    if factor < 0 or factor > 1:
        raise ValueError('Provided factor value is not between 0 and 1. Aborting.')

    # Convert da['time'] to DOY for easier indexing
    da_doy = xr.apply_ufunc(
        lambda time: DOY(time, target_year),
        da['time'],
        vectorize=True
    )
    da = da.assign_coords(time=da_doy).transpose('time', 'y', 'x')

    # Initialize output arrays
    da_eos_times1 = xr.full_like(da_peak_times1, np.nan, dtype='float32')
    da_eos_values1 = xr.full_like(da_peak_times1, np.nan, dtype='float32')
    da_eos_times2 = xr.full_like(da_peak_times1, np.nan, dtype='float32')
    da_eos_values2 = xr.full_like(da_peak_times1, np.nan, dtype='float32')

    #print('Beginning to calculate the end of the first season.')
    eos_target_value1 = da_valley_r_values1 + da_amplitude_r_values1 * factor

    # Find the EOS1 point in the data array (first peak)
    da_eos_times1 = xr.where((da.time >= da_peak_times1) & (da.time <= da_valley_r_times1) &
                             (da <= eos_target_value1), da.time, np.nan).min(dim='time')
    da_eos_times1 = xr.where((da_eos_times1 >= da.time.min()) & (da_eos_times1 <= da.time.max()),
                             da_eos_times1, np.nan)
    # Interpolating da to match the identified sos times
    #da_eos_values1 = da.sel(time=da_eos_times1, method='nearest')
    eos_indices1 = da.time == da_eos_times1
    da_eos_values1 = da.where(eos_indices1, drop=False).isel(time=0)
    #print('> Success!\n')

    #print('Beginning to calculate the end of the second season.')
    # Initialize sos2 as None for cases without a second peak
    da_eos_times2, da_eos_values2 = None, None

    # Check and calculate SOS for second peak if available
    if da_valley_r_times2 is not None and da_amplitude_r_values2 is not None:
        eos_target_value2 = da_valley_r_values2 + da_amplitude_r_values2 * factor
        da_eos_times2 = xr.where((da.time >= da_peak_times2) & (da.time <= da_valley_r_times2) &
                                 (da <= eos_target_value2), da.time, np.nan).min(dim='time')
        da_eos_times1 = xr.where((da_eos_times1 >= da.time.min()) & (da_eos_times1 <= da.time.max()),
                                da_eos_times1, np.nan)
        # Interpolating da to match the identified sos times for the second peak
        #da_eos_values2 = da.sel(time=da_eos_times2, method='nearest')
        eos_indices2 = da.time == da_eos_times2
        da_eos_values2 = da.where(eos_indices2, drop=False).isel(time=0)

    # 如果 da_eos_times1 是 NaN，则相应的 da_eos_values1 也设为 NaN
    eos_times1_nan_mask = np.isnan(da_eos_times1)
    da_eos_values1 = xr.where(~eos_times1_nan_mask, da_eos_values1, np.nan)

    # 如果 da_sos_times2 是 NaN，则相应的 da_sos_values2 也设为 NaN
    if da_eos_times2 is not None:
        eos_times2_nan_mask = np.isnan(da_eos_times2)
        da_eos_values2 = xr.where(~eos_times2_nan_mask, da_eos_values2, np.nan)
        
    da_eos_times1 = da_eos_times1.astype('float32').rename('eos_times1')
    da_eos_values1 = da_eos_values1.astype('float32').rename('eos_values1')
    da_eos_times2 = da_eos_times2.astype('float32').rename('eos_times2')
    da_eos_values2 = da_eos_values2.astype('float32').rename('eos_values2')
    #print('> Success!\n')

    return da_eos_times1, da_eos_values1, da_eos_times2, da_eos_values2


def get_los(da_sos_times1, da_eos_times1, da_sos_times2, da_eos_times2):
    """
    Takes two xarray DataArrays containing the start of season (sos) times (day of year)
    and end of season (eos) times (day of year) and calculates the length of season (los).
    This is calculated as eos day of year minus sos day of year per pixel.

    Parameters
    ----------
    da : xarray DataArray
        A two-dimensional or multi-dimensional array containing an DataArray of veg_index
        and time values.
    da_sos_times: xarray DataArray
        An xarray DataArray type with an x and y dimension (no time). Each pixel is the
        time (day of year) detected at start of season (sos).
    da_eos_times: xarray DataArray
        An xarray DataArray type with an x and y dimension (no time). Each pixel is the
        time (day of year) detected at end of season (eos).

    Returns
    -------
    da_los_values : xarray DataArray
        An xarray DataArray type with an x and y dimension (no time). Each pixel is the
        length of season (los) value detected between the sos and eos day of year values
        across the timeseries at each pixel. The values in los represents number of days.
    """

    # notify user
    #print('Beginning to calculate the length of season (los) values (times not possible).')

    da_los_values1 = da_eos_times1 - da_sos_times1
    da_los_values2 = da_eos_times2 - da_sos_times2

    # Replace negative values with NaN
    da_los_values1 = xr.where(da_los_values1 < 0, np.nan, da_los_values1)
    da_los_values2 = xr.where(da_los_values2 < 0, np.nan, da_los_values2)

    da_los_values1 = da_los_values1.astype('float32').rename('los_values1')
    da_los_values2 = da_los_values2.astype('float32').rename('los_values2')

    #print('> Success!\n')

    return da_los_values1, da_los_values2



def calc_phenometrics(da, target_year, factor=0.5, n_jobs=100):
    # notify user
    #print('Initialising calculation of phenometrics.\n')
    
    # check if dask - not yet supported
    if dask.is_dask_collection(da):
        raise TypeError('Dask arrays not yet supported. Please compute first.')
    
    # check if dataset type
    if type(da) != xr.DataArray:
        raise TypeError('> Not a data array. Please provide a xarray data array.')
        
    # get crs info before work
    #crs = extract_crs(da=da)
    
    # take a mask of all-nan slices for clean up at end and set all-nan to 0s
    da_all_nan_mask = da.isnull().all('time')
    da_all_zero_mask = (da==0).all('time')
    combined_mask = da_all_nan_mask | da_all_zero_mask
    da = da.where(~combined_mask, 0.0)
    
    # notify user
    #print('Beginning calculation of phenometrics. This can take awhile - please wait.\n')
    
    # calc mean value of season
    da_mean_values = get_mean(da=da)

    # calc peak values and times of seasons, and number of seasons
    da_nos, da_pos_times1, da_pos_values1, da_pos_times2, da_pos_values2 = get_pos(da=da, 
                                                                                   mean_values=da_mean_values,
                                                                                   target_year=target_year,
                                                                                   n_jobs=n_jobs)  
    
    # calc valley of season (vos) values and times
    da_vos_l_times1, da_vos_l_values1, da_vos_r_times1, da_vos_r_values1,\
    da_vos_l_times2, da_vos_l_values2, da_vos_r_times2, da_vos_r_values2 = get_vos(da=da, target_year=target_year, 
                                                                                    da_peak_times1=da_pos_times1, 
                                                                                    da_peak_times2=da_pos_times2,
                                                                                    da_peak_values1=da_pos_values1, 
                                                                                    da_peak_values2=da_pos_values2,
                                                                                    n_jobs=n_jobs)

    # update the pos and vos
    da_pos_times1, da_pos_values1, da_vos_l_times1, da_vos_l_values1, da_vos_r_times1, da_vos_r_values1,\
    da_pos_times2, da_pos_values2, da_vos_l_times2, da_vos_l_values2, da_vos_r_times2, da_vos_r_values2,\
    da_nos = get_valid_cycle(da_pos_times1, da_pos_values1, da_vos_l_times1, da_vos_l_values1, da_vos_r_times1, da_vos_r_values1,
                             da_pos_times2, da_pos_values2, da_vos_l_times2, da_vos_l_values2, da_vos_r_times2, da_vos_r_values2,
                             da_nos=da_nos)
    
    # calc base (bse) values (time not possible).
    da_bse_values1, da_bse_values2 = get_bse(da_valley_l_values1=da_vos_l_values1,
                                             da_valley_r_values1=da_vos_r_values1,
                                             da_valley_l_values2=da_vos_l_values2,
                                             da_valley_r_values2=da_vos_r_values2)

   
    # calc amplitude of season (aos) values (time not possible).
    da_aos_l_values1, da_aos_r_values1, da_aos_values1, \
    da_aos_l_values2, da_aos_r_values2, da_aos_values2 = get_aos(da_peak_values1=da_pos_values1,
                                                                 da_valley_l_values1=da_vos_l_values1,
                                                                 da_valley_r_values1=da_vos_r_values1,
                                                                 da_peak_values2=da_pos_values2,
                                                                 da_valley_l_values2=da_vos_l_values2,
                                                                 da_valley_r_values2=da_vos_r_values2)
    
    
    # calc start of season (sos) values and times. takes peak, base metrics and factor
    da_sos_times1, da_sos_values1, da_sos_times2, da_sos_values2 = get_sos(da=da, target_year=target_year,
                                                                           da_valley_l_times1=da_vos_l_times1,
                                                                           da_valley_l_values1=da_vos_l_values1,
                                                                           da_amplitude_l_values1=da_aos_l_values1,
                                                                           da_peak_times1=da_pos_times1,
                                                                           da_valley_l_times2=da_vos_l_times2,
                                                                           da_valley_l_values2=da_vos_l_values2,
                                                                           da_amplitude_l_values2=da_aos_l_values2,
                                                                           da_peak_times2=da_pos_times2,
                                                                           factor=0.5)


    # calc end of season (eos) values and times. takes peak, base metrics and factor
    da_eos_times1, da_eos_values1, da_eos_times2, da_eos_values2 = get_eos(da=da, target_year=target_year,
                                                                           da_valley_r_times1=da_vos_r_times1,
                                                                           da_valley_r_values1=da_vos_r_values1,
                                                                           da_amplitude_r_values1=da_aos_r_values1,
                                                                           da_peak_times1=da_pos_times1,
                                                                           da_valley_r_times2=da_vos_r_times2,
                                                                           da_valley_r_values2=da_vos_r_values2,
                                                                           da_amplitude_r_values2=da_aos_r_values2,
                                                                           da_peak_times2=da_pos_times2,
                                                                           factor=0.5)

    # calc length of season (los) values (time not possible). takes sos and eos
    da_los_values1, da_los_values2 = get_los(da_sos_times1, da_eos_times1, da_sos_times2, da_eos_times2)
    
    # create data array list
    da_list = [
        da_mean_values,
        da_nos, da_pos_times1, da_pos_values1, da_pos_times2, da_pos_values2,
        da_vos_l_times1, da_vos_l_values1, da_vos_r_times1, da_vos_r_values1,
        da_vos_l_times2, da_vos_l_values2, da_vos_r_times2, da_vos_r_values2,
        da_bse_values1, da_bse_values2,
        da_aos_values1, da_aos_values2,
        # da_aos_l_values1, da_aos_r_values1, da_aos_l_values2, da_aos_r_values2,  
        da_sos_times1, da_sos_times2,
        #da_sos_values1, da_sos_values2,
        da_eos_times1, da_eos_times2,
        #da_eos_values1, da_eos_values2,
        da_los_values1, da_los_values2
    ]
  
    # combine data arrays into one dataset
    ds_phenos = xr.merge(da_list, compat='override')
    
    # set original all nan pixels back to nan
    ds_phenos = ds_phenos.where(~combined_mask)
    
    # notify user
    #print('Phenometrics calculated successfully!')
    
    return ds_phenos

