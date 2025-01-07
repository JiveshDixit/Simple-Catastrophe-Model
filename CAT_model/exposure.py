# -*- coding: utf-8 -*-
# Copyright (c) 2024 Jivesh Dixit, P.S. II, NCMRWF
# All rights reserved.
#
# This software is licensed under MIT license.
# Contact [jdixit@nic.in; jiveshdixit@gmail.com]



###### catastrophe_model/exposure.py ######

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import logging
# from .utils import (
#     calculate_damage_percentage,  # We'll define next
#     calculate_damage_from_precipitation  # We'll define next
# )

def calculate_damage_percentage(
    wind_speed_kph: float,
    wind_gust_kph: float,
    distance_km: float,
    radius_km: float,
    building_type: str
) -> float:
    """
    Calculates combined damage percentage based on wind speed & wind gust.
    """
    if distance_km > radius_km:
        return 0.0


    damage_speed = (wind_speed_kph - 80) / 80 * 100  
    damage_speed = np.clip(damage_speed, 0, 100)

    damage_gust = (wind_gust_kph - 100) / 100 * 100  
    damage_gust = np.clip(damage_gust, 0, 100)

    resilience_factors = {
        'residential': 0.8,
        'commercial': 0.6,
        'industrial': 0.5,
        'retail': 0.6,
        'office': 0.6,
        'other': 0.7
    }
    resilience = resilience_factors.get(building_type.lower(), 0.7)

    adjusted_damage_speed = damage_speed * (1 - resilience)
    adjusted_damage_gust = damage_gust * (1 - resilience)
    combined_damage = (adjusted_damage_speed + adjusted_damage_gust) / 2.0


    combined_damage *= (1 - (distance_km / radius_km))
    return float(np.clip(combined_damage, 0, 100))

def calculate_damage_from_precipitation(
    total_precip_mm: float,
    distance_km: float,
    radius_km: float,
    building_type: str
) -> float:
    """
    Calculate damage percentage based on precipitation.
    """
    if distance_km > radius_km:
        return 0.0

    if total_precip_mm < 20:
        damage = 0.0
    elif total_precip_mm > 100:
        damage = 100.0
    else:
        damage = ((total_precip_mm - 20) / 100) * 100

    damage = np.clip(damage, 0, 100)

    resilience_factors = {
        'residential': 0.7,
        'commercial': 0.6,
        'industrial': 0.5,
        'retail': 0.6,
        'office': 0.6,
        'other': 0.7
    }
    resilience = resilience_factors.get(building_type.lower(), 0.7)
    adjusted_damage = damage * (1 - resilience)

    adjusted_damage *= (1 - (distance_km / radius_km))
    return float(np.clip(adjusted_damage, 0, 100))

def calculate_losses(
    assets_gdf: gpd.GeoDataFrame, 
    cyclone_df: pd.DataFrame, 
    radius_km: float = 160
):
    """
    Calculates financial losses for assets based on wind speed, gust, precipitation.
    """
    loss_records = []

    for _, cyclone in cyclone_df.iterrows():
        cyclone_id = cyclone['Cyclone_ID']
        wind_speed = cyclone['wind_speed_kph']
        wind_gust = cyclone.get('wind_gust_kph', 0)
        total_precip = cyclone.get('total_precipitation', 0)
        lat = cyclone['latitude']
        lon = cyclone['longitude']

        center_point = Point(lon, lat)


        assets_proj = assets_gdf.to_crs(epsg=32614)
        cyclone_point_proj = gpd.GeoSeries([center_point], crs="EPSG:4326").to_crs(epsg=32614).iloc[0]

        assets_proj['Distance_km'] = assets_proj.geometry.distance(cyclone_point_proj) / 1000.0


        assets_proj['Damage_Wind_Speed_%'] = assets_proj.apply(
            lambda row: calculate_damage_percentage(
                wind_speed_kph=wind_speed,
                wind_gust_kph=wind_gust,
                distance_km=row['Distance_km'],
                radius_km=radius_km,
                building_type=row.get('Building_Type', 'other')
            ),
            axis=1
        )


        assets_proj['Damage_Precipitation_%'] = assets_proj.apply(
            lambda row: calculate_damage_from_precipitation(
                total_precip_mm=total_precip,
                distance_km=row['Distance_km'],
                radius_km=radius_km,
                building_type=row.get('Building_Type', 'other')
            ),
            axis=1
        )


        if assets_proj['Damage_Precipitation_%'].any() !=0:
            assets_proj['Combined_Damage_%'] = (
                assets_proj['Damage_Wind_Speed_%'] + assets_proj['Damage_Precipitation_%']
            ) / 2.0
        else:
            assets_proj['Combined_Damage_%'] = assets_proj['Damage_Wind_Speed_%']

        assets_proj['Loss_USD'] = (
            assets_proj.get('Value_USD', 0) * (assets_proj['Combined_Damage_%'] / 100.0)
        ).fillna(0).astype('float32')

        total_loss = float(assets_proj['Loss_USD'].sum())

        loss_records.append({
            'Cyclone_ID': cyclone_id,
            'Wind_Speed_kph': wind_speed,
            'Wind_Gust_kph': wind_gust,
            'Total_Precip_mm': total_precip,
            'Latitude': lat,
            'Longitude': lon,
            'Total_Loss_USD': total_loss
        })


        assets_gdf['Distance_km'] = assets_proj['Distance_km']
        assets_gdf['Damage_Wind_Speed_%'] = assets_proj['Damage_Wind_Speed_%']
        assets_gdf['Damage_Precipitation_%'] = assets_proj['Damage_Precipitation_%']
        assets_gdf['Combined_Damage_%'] = assets_proj['Combined_Damage_%']
        assets_gdf['Loss_USD'] = assets_proj['Loss_USD']

    loss_df = pd.DataFrame(loss_records)
    return loss_df, assets_gdf
