import os, sys
import re
import glob
import numpy as np
import xarray as xr
import pandas as pd
import math
import dask
import dask.array as da
import datacube
import datetime
import time
import rasterio
import rioxarray
import h5py
import gc
from multiprocessing import Pool
from rasterio.windows import Window
from rasterio import open as rio_open
from rasterio.transform import from_origin
from osgeo import gdal,osr

import scipy.sparse
import scipy.signal as signal
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.stats import zscore, kstest, norm
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter, medfilt, wiener
from dask.distributed import Client
from dask.diagnostics import ProgressBar
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import func

# 定义数据处理函数
def process_block(evi2_files, row_off, col_off, block_size, time_range, target_year):
    window = rasterio.windows.Window(col_off=col_off, row_off=row_off, width=block_size, height=block_size)
    
    # 读取该块的所有时间数据
    data_list = []
    for evi2_file in evi2_files:
        with rasterio.open(evi2_file) as src:
            data = src.read(1, window=window)
            data_list.append(data)
    
    data_array = np.stack(data_list, axis=0)  # Shape: (time, height, width)
    
    # 转换为 xarray 数据集
    evi2_ds = xr.Dataset(
        data_vars={"EVI2": (("time", "y", "x"), data_array)},
        coords={
            "time": time_range,
            "y": np.arange(row_off, row_off + block_size),
            "x": np.arange(col_off, col_off + block_size)
        }
    )

    # Step 1: 去除离群值
    evi2_ds = func.remove_outliers(ds=evi2_ds, user_factor=2)
    
    # Step 2: 单样条插值
    evi2_ds_chunked = evi2_ds.chunk({'time': -1}) 
    evi2_data = evi2_ds_chunked['EVI2'].values
    time_values = np.arange(evi2_ds_chunked.dims['time'])
    interpolated_data = func.interpolate_spline(evi2_data, time_values, s=1.0)
    evi2_ds = xr.Dataset(
        data_vars={ "EVI2":(('time','y','x'), interpolated_data.astype(np.float32))}, #interpolated_data
        coords={
            "time": evi2_ds['time'],
            "y": evi2_ds['y'],
            "x": evi2_ds['x']
        }).chunk({'time': -1})

    # Step 3: SG 滤波
    evi2_ds = func.smooth_sg(ds=evi2_ds, window_length=91, polyorder=3)
    
    evi2_ds = evi2_ds.isel(time=slice(182, -184)).chunk({'time': -1})
    
    # Step 4: 提取关键物候参数
    phenometrics = func.calc_phenometrics(da=evi2_ds['EVI2'].compute(), target_year=target_year, factor=0.5)
    
    return phenometrics

# 批量处理年份
def process_years(start_year, end_year, tile, input_folder, output_folder, block_size=20):
    for target_year in range(start_year, end_year + 1):
        start = time.perf_counter()
        print(f"Processing year: {target_year}")

        # 定义数据路径
        EVI2_folder = os.path.join(input_folder, tile)
        all_files = sorted(glob.glob(os.path.join(EVI2_folder, '*.tif')))
        
        # 筛选年份范围的文件
        years_to_select = {str(target_year - 1), str(target_year), str(target_year + 1)}
        evi2_files = [f for f in all_files if os.path.basename(f)[9:13] in years_to_select]
        
        if evi2_files:
            print(f'First data: {os.path.basename(evi2_files[0])}')
            print(f'Last data: {os.path.basename(evi2_files[-1])}')
        print(f'File length: {len(evi2_files)}\n')

        # 时间范围
        firstday = pd.Timestamp(f'{target_year-1}-01-01')
        lastday = pd.Timestamp(f'{target_year+1}-12-31')
        time_range = pd.date_range(start=firstday, end=lastday, freq='D')
        print(f'Time range:\nfrom {time_range[0].strftime("%Y-%m-%d")} to {time_range[-1].strftime("%Y-%m-%d")}, length of time: {len(time_range)}\n')

        # 处理块
        row_offs, col_offs = range(0, 2400, block_size), range(0, 2400, block_size)
        total_blocks = len(row_offs) * len(col_offs)
        
        with tqdm_joblib(desc=f"Year {target_year} Processing", total=total_blocks) as progress_bar:
            results = Parallel(n_jobs=100)(
                delayed(process_block)(
                    evi2_files, row_off, col_off, block_size, time_range, target_year
                ) 
                for row_off in row_offs for col_off in col_offs
            )
        
        # 合并结果
        metric_names = list(results[0].data_vars.keys())
        final_metrics = {name: [] for name in metric_names}

        for result in results:
            for name in metric_names:
                final_metrics[name].append(result[name].values)

        first_file = evi2_files[183]
        with rasterio.open(first_file) as src:
            window = Window(col_off=0, row_off=0, width=2400, height=2400)
            window_transform = src.window_transform(window)
            crs = src.crs

        # 保存结果为 GeoTIFF
        for metric_name, blocks in final_metrics.items():
            merged_data = da.block([[blocks[i * len(col_offs) + j] for j in range(len(col_offs))] for i in range(len(row_offs))])
            output_file_path = os.path.join(output_folder, f'GDPD_{target_year}_{metric_name}_{tile}.tif')
            
            with rasterio.open(
                    output_file_path, 'w', 
                    driver='GTiff',
                    height=merged_data.shape[0],
                    width=merged_data.shape[1],
                    count=1,
                    dtype='float32',
                    crs=crs,
                    transform=window_transform,
                    compress='lzw'
            ) as dst:
                dst.write(merged_data, 1)

            print(f"Raster file for {metric_name} saved successfully to {output_file_path}")

        # 清理内存
        del evi2_files, results, final_metrics
        gc.collect()  # 手动垃圾回收

        end = time.perf_counter()
        runTime = end - start

        if runTime < 60:
            print("运行时间：", f"{runTime:.2f}", "秒")
        elif runTime < 3600:
            minutes = int(runTime // 60)
            seconds = runTime % 60
            print(f"运行时间：{minutes}分{seconds:.2f}秒")
        else:
            hours = int(runTime // 3600)
            minutes = int((runTime % 3600) // 60)
            seconds = runTime % 60
            print(f"{target_year}处理完毕\n运行时间：{hours}小时{minutes}分{seconds:.2f}秒\n")                

# 使用函数处理
process_years(
    start_year=2001, 
    end_year=2020, 
    tile='{tile}', # define the MODIS tile 
    input_folder='{input_folder}',  # define the input direction
    output_folder='{output_folder}'  # define the output direction
)
