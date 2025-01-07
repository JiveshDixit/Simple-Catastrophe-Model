# -*- coding: utf-8 -*-
# Copyright (c) 2024 Jivesh Dixit, P.S. II, NCMRWF
# All rights reserved.
#
# This software is licensed under MIT license.
# Contact [jdixit@nic.in; jiveshdixit@gmail.com]



###### catastrophe_model/utils.py ######

import os
import time
import logging
import uuid
import gc
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box, Point
from requests.exceptions import RequestException
from urllib3.exceptions import ProtocolError
import osmnx as ox
from osmnx._errors import InsufficientResponseError
import datetime
import cdsapi

# Configure OSMnx settings
ox.settings.use_cache = True
ox.settings.log_console = False  
ox.settings.timeout = 60  

# Define Overpass API servers
OVERPASS_SERVERS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter"
]

def split_polygon_into_grid(gdf: gpd.GeoDataFrame, grid_size_km: float = 50) -> gpd.GeoDataFrame:
    """
    Splits a GeoDataFrame polygon into a grid of smaller polygons.
    """
    gdf_proj = gdf.to_crs(epsg=32614)  # UTM Zone 14N (converts to distance in kilo-meters)
    minx, miny, maxx, maxy = gdf_proj.total_bounds
    grid_size_m = grid_size_km * 1000

    grid_cells = []
    x = minx
    while x < maxx:
        y = miny
        while y < maxy:
            grid_cells.append(box(x, y, x + grid_size_m, y + grid_size_m))
            y += grid_size_m
        x += grid_size_m

    grid = gpd.GeoDataFrame({'geometry': grid_cells}, crs=gdf_proj.crs)
    grid_clipped = gpd.overlay(grid, gdf_proj, how='intersection')
    grid_clipped.sindex
    return grid_clipped.to_crs(epsg=4326)  # Back to WGS84 or lat-lon

def _fetch_buildings_for_polygon(
    grid_polygon: Point, 
    idx: int, 
    max_retries: int = 3, 
    sleep_time: int = 5
) -> gpd.GeoDataFrame:
    """
    Fetch building footprints within a single grid polygon using OSMnx.
    Retries on transient errors, rotating Overpass servers.
    """
    buildings = gpd.GeoDataFrame()
    server_index = 0
    num_servers = len(OVERPASS_SERVERS)

    for attempt in range(1, max_retries + 1):
        try:
            ox.settings.overpass_endpoint = OVERPASS_SERVERS[server_index]
            logging.info(f"Grid Cell {idx}: Using Overpass server: {OVERPASS_SERVERS[server_index]}")

            buildings = ox.features.features_from_polygon(
                grid_polygon, 
                tags={
                    "building": ['residential', 'commercial', 'industrial', 'retail', 'office', 'other']
                    }
            )

            if not buildings.empty:
                logging.info(f"Grid Cell {idx}: Successfully fetched {len(buildings)} buildings.")
            else:
                logging.info(f"Grid Cell {idx}: No matching features found.")
            break
        except (RequestException, ProtocolError, InsufficientResponseError) as e:
            logging.error(f"Grid Cell {idx} Attempt {attempt} failed: {e}")
            if attempt < max_retries:
                server_index = (server_index + 1) % num_servers
                logging.info(f"Retrying with a different server in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                logging.error(f"Grid Cell {idx}: Max retries reached. Skipping.")
    return buildings

def optimize_data_types(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Optimize data types of a GeoDataFrame to save memory.
    """
    float_cols = ['Value_USD', 'Latitude', 'Longitude', 'Distance_km',
                  'Damage_%', 'Loss_USD']
    for col in float_cols:
        if col in gdf.columns:
            gdf[col] = pd.to_numeric(gdf[col], errors='coerce').astype('float32')

    category_cols = ['Building_Type']
    for col in category_cols:
        if col in gdf.columns and pd.api.types.is_categorical_dtype(gdf[col]):
            gdf[col] = gdf[col].astype(str)

    return gdf

def sanitize_columns(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Cleans and shortens column names to avoid conflicts with GeoPackage.
    """
    gdf.columns = [col[:63] for col in gdf.columns]  # Truncate to 63 characters
    gdf.columns = [col.replace(":", "_").replace("-", "_").replace(" ", "_")
                  for col in gdf.columns]
    gdf = gdf.loc[:, ~gdf.columns.duplicated()]
    return gdf

def save_to_geopackage(gdf: gpd.GeoDataFrame, filename: str, layer: str = "buildings"):
    """
    Saves a GeoDataFrame to a GeoPackage, ensuring compatibility.
    """
    try:
        # If needed, handle columns like 'FIXME'
        if 'FIXME' in gdf.columns:
            gdf.rename(columns={'FIXME': 'FIXME_notes'}, inplace=True)
            gdf['FIXME_notes'] = gdf['FIXME_notes'].astype(str)

        # Convert categorical columns to string to avoid errors
        for col in gdf.columns:
            if pd.api.types.is_categorical_dtype(gdf[col]):
                gdf[col] = gdf[col].astype(str)

        gdf = sanitize_columns(gdf)
        gdf.to_file(filename, driver='GPKG', layer=layer, encoding='UTF-8')
        logging.info(f"GeoDataFrame saved to {filename}")
    except Exception as e:
        logging.error(f"Error saving GeoDataFrame to GeoPackage: {e}")

def download_era5_data(
    cyclone_name: str,
    north: float, 
    south: float, 
    east: float, 
    west: float,
    start_date: str, 
    end_date: str,
    max_retries: int = 3, 
    sleep_time: int = 5
) -> list:
    """
    Downloads ERA5 data for the specified cyclone period using CDS API.
    """
    c = cdsapi.Client(verify=False)

    start_dt = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    num_days = (end_dt - start_dt).days + 1
    days = [(start_dt + datetime.timedelta(days=i)).day for i in range(num_days)]
    days_str = [f"{day:02d}" for day in days]

    month_str_let = start_dt.strftime('%b')  
    month_str = start_dt.strftime('%m')      

    filepaths = [
        f'era5_{cyclone_name}_{month_str_let}_{days_str[0]}-{days_str[-1]}_{start_dt.year}_extended.nc',
        f'era5_{cyclone_name}_{month_str_let}_{days_str[0]}-{days_str[-1]}_{start_dt.year}_extended_tot_precip.nc'
    ]

    variables = [
        [
            '10m_u_component_of_wind',
            '10m_v_component_of_wind',
            'mean_sea_level_pressure',
            'instantaneous_10m_wind_gust'  
        ],
        ['tp']
    ]

    final_filepaths = []

    for var_set, filepath in zip(variables, filepaths):
        if not os.path.exists(filepath):
            logging.info(f"File '{filepath}' not found. Initiating download...")
            attempt = 0
            while attempt < max_retries:
                try:
                    c.retrieve(
                        'reanalysis-era5-single-levels',
                        {
                            'product_type': 'reanalysis',
                            'format': 'netcdf',
                            'variable': var_set,
                            'year': start_dt.year,
                            'month': month_str,
                            'day': days_str,
                            'time': [
                                "00:00", "01:00", "02:00",
                                "03:00", "04:00", "05:00",
                                "06:00", "07:00", "08:00",
                                "09:00", "10:00", "11:00",
                                "12:00", "13:00", "14:00",
                                "15:00", "16:00", "17:00",
                                "18:00", "19:00", "20:00",
                                "21:00", "22:00", "23:00"
                            ],
                            'area': [
                                north, west, south, east
                            ],
                        },
                        filepath
                    )
                    logging.info(f"Successfully downloaded '{filepath}'.")
                    break
                except Exception as e:
                    attempt += 1
                    logging.error(f"Attempt {attempt} failed for '{filepath}': {e}")
                    if attempt < max_retries:
                        logging.info(f"Retrying in {sleep_time} seconds...")
                        time.sleep(sleep_time)
                    else:
                        logging.error(f"Failed to download '{filepath}' after {max_retries} attempts.")
        else:
            logging.info(f"File '{filepath}' already exists. Skipping download.")
        
        final_filepaths.append(filepath)
    
    logging.info("ERA5 data download process completed.")
    return final_filepaths

def print_cyclone_summary(landfall_df: pd.DataFrame):
    """
    Prints a summary of the cyclone at landfall.
    """
    if landfall_df.empty:
        logging.warning("Landfall DataFrame is empty. No summary to print.")
        return

    row = landfall_df.iloc[0]
    summary = (
        "\n--- Landfall Summary ---\n"
        f"Method of Detection  : {row.get('method', 'N/A')}\n"
        f"Wind Speed (kph)     : {row.get('wind_speed_kph', 'N/A'):.1f}\n"
        f"Wind Gust (kph)      : {row.get('wind_gust_kph', 'N/A'):.1f}\n"
        f"MSL (hPa)            : {row.get('msl_hPa', 'N/A'):.1f}\n"
        f"Total Precipitation  : {row.get('total_precipitation', 'N/A'):.1f} mm\n"
        f"Latitude, Longitude  : {row.get('latitude', 'N/A'):.2f}, {row.get('longitude', 'N/A'):.2f}\n"
        f"Time                 : {row.get('time', 'N/A')}\n"
        "------------------------"
    )
    print(summary)
    logging.info("Cyclone summary printed.")


def write_loss_data_to_csv(updated_assets_gdf: pd.DataFrame, 
                            output_filename: str):
    """
    Writes cyclone loss data to a CSV file, including building type information.
    Includes only rows where "Loss_USD" is greater than 0.

    Args:
        loss_df (pd.DataFrame): DataFrame containing loss data with a 'Building_Type' column.
        output_filename (str): Name of the output CSV file.
    """

    try:

        required_columns = ["Building_Type", "Building_ID", "addr_city", "Value_USD", "Damage_Wind_Speed_%", \
                            "Damage_Precipitation_%", "Combined_Damage_%", "Loss_USD"]  # Add other relevant columns
        if not all(col in updated_assets_gdf.columns for col in required_columns):
            raise ValueError("Loss DataFrame is missing required columns.")


        updated_assets_gdf = updated_assets_gdf[required_columns]
        updated_assets_gdf = updated_assets_gdf[updated_assets_gdf["Loss_USD"] > 0]


        updated_assets_gdf.to_csv(output_filename, index=False)  # Set index=False to exclude row numbers
        print(f"Loss data written to {output_filename}")

    except Exception as e:
        print(f"Error writing loss data to CSV: {e}")