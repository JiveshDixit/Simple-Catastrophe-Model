# -*- coding: utf-8 -*-
# Copyright (c) 2024 Jivesh Dixit, P.S. II, NCMRWF
# All rights reserved.
#
# This software is licensed under MIT license.
# Contact [jdixit@nic.in; jiveshdixit@gmail.com]


###### catastrophe_model/hazard.py ######

import os
import time
import datetime
import numpy as np
import pandas as pd
import xarray as xr
import logging
from utils import (
    print_cyclone_summary,
    download_era5_data,
)

LANDFALL_DETECTION_METHOD = 'max_wind'

def open_and_prepare_dataset(
    cyclone_name: str,
    north: float, 
    south: float, 
    east: float, 
    west: float,
    start_date: str, 
    end_date: str,
    filepath_1: str = None, 
    filepath_2: str = None
) -> xr.Dataset:
    """
    Opens ERA5 NetCDF files, processes them, and calculates additional variables.
    If files are missing, downloads them.
    """
    if filepath_1 is not None and filepath_2 is not None:
        missing_files = []
        if not os.path.exists(filepath_1):
            missing_files.append(filepath_1)
        if not os.path.exists(filepath_2):
            missing_files.append(filepath_2)
        if missing_files:
            logging.info(f"Missing files: {missing_files}. Downloading ERA5 data...")
            downloaded_files = download_era5_data(
                cyclone_name=cyclone_name,
                north=north, south=south, east=east, west=west,
                start_date=start_date, end_date=end_date
            )
            filepath_1, filepath_2 = downloaded_files
    else:
        downloaded_files = download_era5_data(
            cyclone_name=cyclone_name,
            north=north, south=south, east=east, west=west,
            start_date=start_date, end_date=end_date
        )
        filepath_1, filepath_2 = downloaded_files

    if not os.path.exists(filepath_1):
        raise FileNotFoundError(f"ERA5 file '{filepath_1}' not found after download.")
    if not os.path.exists(filepath_2):
        raise FileNotFoundError(f"ERA5 precipitation file '{filepath_2}' not found after download.")

    logging.info(f"Opening ERA5 data files: '{filepath_1}' and '{filepath_2}'.")
    ds_1 = xr.open_dataset(filepath_1, chunks={'time': 10})
    ds_2 = xr.open_dataset(filepath_2, chunks={'time': 10})

    if 'valid_time' in ds_1.coords:
        ds_1 = ds_1.rename({'valid_time': 'time'})
    if 'valid_time' in ds_2.coords:
        ds_2 = ds_2.rename({'valid_time': 'time'})

    required_vars = ['u10', 'v10']
    for var in required_vars:
        if var not in ds_1.data_vars:
            raise ValueError(f"Missing '{var}' in the ERA5 dataset.")
    
    ds_1['wind_speed_mps'] = np.sqrt(ds_1['u10']**2 + ds_1['v10']**2)
    ds_1['wind_speed_kph'] = ds_1['wind_speed_mps'] * 3.6

    if 'i10fg' in ds_1.data_vars:
        ds_1['wind_gust_kph'] = ds_1['i10fg'] * 3.6
    elif '10si' in ds_1.data_vars:
        ds_1['wind_gust_kph'] = ds_1['10si'] * 3.6
    else:
        ds_1['wind_gust_kph'] = np.nan
        logging.warning("No wind gust variable found, 'wind_gust_kph' set to NaN.")
    
    if 'msl' in ds_1.data_vars:
        ds_1['msl_hPa'] = ds_1['msl'] / 100.0
    else:
        ds_1['msl_hPa'] = np.nan
        logging.warning("No MSL variable found, 'msl_hPa' set to NaN.")

    if 'tp' in ds_2.data_vars:
        ds_1['total_precipitation'] = ds_2['tp'] * 1000
    else:
        ds_1['total_precipitation'] = np.nan
        logging.warning("No total_precipitation variable found, set to NaN.")
    
    logging.info("ERA5 datasets processed successfully.")
    return ds_1

def identify_landfall(
    df: pd.DataFrame,
    method: str = 'max_wind'
) -> pd.DataFrame:
    """
    Identifies the landfall event from the ERA5 DataFrame based on 'max_wind' or 'min_mslp'.
    """
    if 'time' not in df.columns:
        raise KeyError("No 'time' column found in DataFrame.")
    
    grouped = df.groupby('time', group_keys=True)

    if method == 'max_wind':
        try:
            maxwind_indices = grouped['wind_speed_kph'].idxmax()
            landfall_event = df.loc[maxwind_indices].sort_values('wind_speed_kph', ascending=False).iloc[0]
        except KeyError as e:
            logging.error(f"Error in idxmax: {e}")
            raise
    elif method == 'min_mslp':
        try:
            minmslp_indices = grouped['msl_hPa'].idxmin()
            landfall_event = df.loc[minmslp_indices].sort_values('msl_hPa', ascending=True).iloc[0]
        except KeyError as e:
            logging.error(f"Error in idxmin: {e}")
            raise
    else:
        raise ValueError("method must be either 'max_wind' or 'min_mslp'")

    landfall_df = pd.DataFrame([{
        'Cyclone_ID': 1,
        'method': method,
        'wind_speed_kph': landfall_event['wind_speed_kph'],
        'wind_gust_kph': landfall_event.get('wind_gust_kph', np.nan),
        'msl_hPa': landfall_event.get('msl_hPa', np.nan),
        'total_precipitation': landfall_event.get('total_precipitation', np.nan),
        'latitude': landfall_event['latitude'],
        'longitude': landfall_event['longitude'],
        'time': landfall_event['time']
    }])

    required_cols = ['wind_speed_kph', 'wind_gust_kph', 'total_precipitation', 'latitude', 'longitude', 'time']
    missing_cols = [c for c in required_cols if c not in landfall_df.columns]
    if missing_cols:
        logging.error(f"Missing columns in landfall_df: {missing_cols}")
        raise KeyError(f"Missing columns: {missing_cols}")

    logging.info("Landfall identified successfully.")
    return landfall_df
