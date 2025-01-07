# -*- coding: utf-8 -*-
# Copyright (c) 2024 Jivesh Dixit, P.S. II, NCMRWF
# All rights reserved.
#
# This software is licensed under MIT license.
# Contact [jdixit@nic.in; jiveshdixit@gmail.com]


###### catastrophe_model/plotting.py ######

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import logging
import folium

def plot_cyclone_features_at_landfall_point(df: pd.DataFrame, landfall_df: pd.DataFrame, cyclone_name:str):
    """
    Plots wind speed, wind gust, and MSLP over time at the specific grid point.
    """
    sns.set(style="whitegrid")
    plt.figure(figsize=(16, 18))

    landfall_lat = landfall_df.iloc[0]['latitude']
    landfall_lon = landfall_df.iloc[0]['longitude']

    landfall_grid_df = df[(df['latitude'] == landfall_lat) & (df['longitude'] == landfall_lon)]
    if landfall_grid_df.empty:
        logging.warning("No data for the landfall grid point. Skipping plot.")
        return

    landfall_grid_df = landfall_grid_df.sort_values(by='time')

    # Wind Speed
    plt.subplot(4, 1, 1)
    sns.lineplot(data=landfall_grid_df, x='time', y='wind_speed_kph', color='blue', label='Wind Speed (kph)', lw=2)
    plt.scatter(
        landfall_df['time'],
        landfall_df['wind_speed_kph'],
        color='red',
        zorder=5,
        label='Landfall'
    )
    plt.title("Wind Speed over Time at Landfall Grid Point")
    plt.xlabel("Time")
    plt.ylabel("Wind Speed (kph)")
    plt.legend()

    # Wind Gust
    plt.subplot(4, 1, 2)
    if 'wind_gust_kph' in landfall_grid_df.columns and landfall_grid_df['wind_gust_kph'].notna().any():
        sns.lineplot(data=landfall_grid_df, x='time', y='wind_gust_kph', color='orange', label='Wind Gust (kph)', lw=2)
        plt.scatter(
            landfall_df['time'],
            landfall_df['wind_gust_kph'],
            color='red',
            zorder=5,
            label='Landfall'
        )
        plt.title("Wind Gust over Time at Landfall Grid Point")
        plt.xlabel("Time")
        plt.ylabel("Wind Gust (kph)")
        plt.legend()
    else:
        plt.text(0.5, 0.5, "No Wind Gust Data Available", ha='center', va='center')
        plt.axis('off')

    # Total Precipitation
    plt.subplot(4, 1, 3)
    if 'total_precipitation' in landfall_grid_df.columns and landfall_grid_df['total_precipitation'].notna().any():
        sns.lineplot(data=landfall_grid_df, x='time', y='total_precipitation', color='orange', label='Total Precipitation (mm)', lw=2)
        plt.scatter(
            landfall_df['time'],
            landfall_df['total_precipitation'],
            color='darkorange',
            zorder=5,
            label='Landfall'
        )
        plt.title("Total Precipitation over Time at Landfall Grid Point")
        plt.xlabel("Time")
        plt.ylabel("Total Precipitation (mm)")
        plt.legend()
    else:
        plt.text(0.5, 0.5, "No Total Precipitation Data Available", ha='center', va='center')
        plt.axis('off')

    # MSLP
    plt.subplot(4, 1, 4)
    if 'msl_hPa' in landfall_grid_df.columns and landfall_grid_df['msl_hPa'].notna().any():
        sns.lineplot(data=landfall_grid_df, x='time', y='msl_hPa', color='green', label='MSL (hPa)', lw=2)
        plt.scatter(
            landfall_df['time'],
            landfall_df['msl_hPa'],
            color='red',
            zorder=5,
            label='Landfall'
        )
        plt.title("MSL over Time at Landfall Grid Point")
        plt.xlabel("Time")
        plt.ylabel("MSL (hPa)")
        plt.legend()
    else:
        plt.text(0.5, 0.5, "No MSL Data Available", ha='center', va='center')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(f'{cyclone_name}_landfall_features.png', bbox_inches='tight')
    logging.info(f'{cyclone_name}_landfall_features.png')
    plt.show()

def create_loss_summary_plot(loss_df: pd.DataFrame, cyclone_name:str):
    """
    Creates and displays a summary plot of total losses.
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(data=loss_df, x='Cyclone_ID', y='Total_Loss_USD', palette='viridis', width=0.6)
    plt.xticks(rotation=45)
    plt.ylabel('Total Loss (USD)')
    plt.xlabel('Cyclone ID')
    plt.title('Total Insured Losses per Cyclone')
    plt.tight_layout()
    plt.savefig(f'{cyclone_name}_total_loss_summary.png', bbox_inches='tight')
    logging.info(f'{cyclone_name}_total_loss_summary.png')
    plt.show()

# def create_detailed_loss_summary_plot(loss_df: pd.DataFrame, cyclone_name:str):
#     """
#     Creates detailed plots of wind speed and precipitation for each cyclone.
#     """
#     plt.figure(figsize=(14, 6))

#     # 1) Wind Speed
#     plt.subplot(1, 2, 1)
#     sns.barplot(data=loss_df, x='Cyclone_ID', y='Wind_Gust_kph', palette='Reds')
#     plt.xticks(rotation=45)
#     plt.ylabel('Wind Speed (kph)')
#     plt.xlabel('Cyclone ID')
#     plt.title(f'Wind Speed {loss_df['wind_gust_kph']}')

#     # 2) Precipitation
#     plt.subplot(1, 2, 2)
#     sns.barplot(data=loss_df, x='Cyclone_ID', y='Total_Precip_mm', palette='Blues')
#     plt.xticks(rotation=45)
#     plt.ylabel('Total Precipitation (mm)')
#     plt.xlabel('Cyclone ID')
#     plt.title(f'Total Precipitation {loss_df['total_precipitation']}')

#     plt.tight_layout()
#     plt.savefig(f'{cyclone_name}_detailed_loss_summary.png', bbox_inches='tight')
#     logging.info(f'{cyclone_name}_detailed_loss_summary.png')
#     plt.show()


def create_interactive_map(cyclone_name: str,
    assets_gdf: gpd.GeoDataFrame, 
    cyclone_df: pd.DataFrame, 
    radius_km: float = 160
):
    """
    Creates an interactive Folium map showing asset losses and cyclone impact.
    """
    # For each row in cyclone_df (for each cyclone), create a separate Folium map
    for _, cyclone in cyclone_df.iterrows():
        lat = cyclone.get('latitude', None)
        lon = cyclone.get('longitude', None)
        if pd.isna(lat) or pd.isna(lon):
            logging.warning("Cyclone center row missing lat/lon. Skipping map creation.")
            continue

        wind_speed = cyclone.get('wind_speed_kph', 0)
        wind_gust = cyclone.get('wind_gust_kph', 0)
        total_precip = cyclone.get('Total_Precip_mm', 0)
        cyclone_id = cyclone.get('Cyclone_ID', 'N/A')
        total_loss = cyclone.get('Total_Loss_USD', 0.0)

        logging.info(f"Creating map for Cyclone ID: {cyclone_id} with Total Loss: {total_loss}")

        # Folium map centered near the cyclone's lat/lon
        m = folium.Map(location=[lat, lon], zoom_start=7)


        folium.Circle(
            location=[lat, lon],
            radius=radius_km * 1000,  # Convert km to meters
            color='blue',
            fill=True,
            fill_opacity=0.1,
            popup=(
                f"Cyclone ID: {cyclone_id}<br>"
                f"Wind Speed: {wind_speed:.1f} kph<br>"
                f"Wind Gust: {wind_gust:.1f} kph<br>"
                f"Total Precipitation: {total_precip:.1f} mm<br>"
                f"Total Loss: ${total_loss:,.2f}"
            )
        ).add_to(m)


        folium.Marker(
            location=[lat, lon],
            popup=f"Cyclone Center ID: {cyclone_id}",
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(m)

        # Loop through each building in assets_gdf
        building_count = 0
        for _, asset in assets_gdf.iterrows():
            asset_lat = asset.get('Latitude', None)
            asset_lon = asset.get('Longitude', None)

            # Skip if lat/lon is missing or NaN
            if pd.isna(asset_lat) or pd.isna(asset_lon):
                continue

            loss = asset.get('Loss_USD', 0.0)
            color = 'red' if loss > 0 else 'green'

            folium.CircleMarker(
                location=[asset_lat, asset_lon],
                radius=3,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.6,
                popup=(
                    f"Building ID: {asset.get('Building_ID', 'N/A')}<br>"
                    f"Type: {asset.get('Building_Type', 'unknown')}<br>"
                    f"Value: ${asset.get('Value_USD', 0):,.2f}<br>"
                    f"Combined Damage: {asset.get('Combined_Damage_%', 0):.1f}%<br>"
                    f"Loss: ${loss:,.2f}"
                )
            ).add_to(m)
            building_count += 1

        logging.info(f"Plotted {building_count} building markers on the map for Cyclone ID {cyclone_id}")

        # Add legend
        legend_html = '''
         <div style="position: fixed; 
                     bottom: 50px; left: 50px; width: 220px; height: 120px; 
                     border:2px solid grey; z-index:9999; font-size:14px;
                     background-color:white;">
         &nbsp;<b>Legend</b><br>
         &nbsp;<i class="fa fa-circle" style="color:red"></i>&nbsp;Buildings with Loss<br>
         &nbsp;<i class="fa fa-circle" style="color:green"></i>&nbsp;No Loss<br>
         &nbsp;<i class="fa fa-circle" style="color:blue"></i>&nbsp;Cyclone Impact Area
         </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))

        # Save the map to an HTML file
        map_filename = f"hurricane_{cyclone_name}_map.html"
        m.save(map_filename)
        logging.info(f"Interactive map saved as '{map_filename}'.\n")
