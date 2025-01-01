import os
import uuid
import time
import logging
import gc
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import osmnx as ox
from shapely.geometry import box, Point
from requests.exceptions import RequestException
from urllib3.exceptions import ProtocolError
from osmnx._errors import InsufficientResponseError  # Use with caution
import concurrent.futures

# (Optional) for ERA5 download via CDS
try:
    import cdsapi
except ImportError:
    pass

# -------------------------------
# 0. Memory Profiling (Optional)
# -------------------------------
# To use memory profiling, uncomment the following lines and decorate functions with @profile
# from memory_profiler import profile

# -------------------------------
# 1. Configuration and Settings
# -------------------------------

OVERPASS_SERVERS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.openstreetmap.ru/api/interpreter"
]

# Configure OSMnx settings
ox.settings.use_cache = True
ox.settings.log_console = False  # Disable console logging to save memory
ox.settings.timeout = 60  # Set a reasonable timeout for OSM queries

# -------------------------------
# 2. Utility Functions
# -------------------------------

def split_polygon_into_grid(gdf, grid_size_km=50):
    """
    Splits a GeoDataFrame polygon into a grid of smaller polygons.

    Args:
        gdf: GeoDataFrame with the polygon (Texas).
        grid_size_km: Size of each grid cell in kilometers.
    """
    # Project to a suitable CRS (e.g., UTM zone for Texas)
    gdf_proj = gdf.to_crs(epsg=32614)  # UTM zone 14N

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
    grid_clipped.sindex  # Trigger spatial indexing
    return grid_clipped.to_crs(epsg=4326)  # Back to WGS84


def _fetch_buildings_for_polygon(grid_polygon, idx, max_retries=3, sleep_time=5):
    """
    Fetch building footprints within a single grid polygon using OSMnx.
    Retries on transient errors; rotates between multiple Overpass servers.
    """
    buildings = gpd.GeoDataFrame()
    server_index = 0  # Start with the first server
    num_servers = len(OVERPASS_SERVERS)

    for attempt in range(1, max_retries + 1):
        try:
            # Set the current Overpass server
            ox.settings.overpass_endpoint = OVERPASS_SERVERS[server_index]
            print(f"Grid Cell {idx}: Using Overpass server: {OVERPASS_SERVERS[server_index]}")

            # Fetch building data with limited fields to save memory
            buildings = ox.features.features_from_polygon(
                grid_polygon, 
                tags={'building': True}
            )

            # Exit loop on success
            if not buildings.empty:
                print(f"Grid Cell {idx}: Successfully fetched {len(buildings)} buildings.")
            else:
                print(f"Grid Cell {idx}: No matching features found.")
            break
        except (RequestException, ProtocolError, InsufficientResponseError) as e:
            print(f"Grid Cell {idx} Attempt {attempt} failed: {e}")
            if attempt < max_retries:
                # Rotate to the next server
                server_index = (server_index + 1) % num_servers
                print(f"Retrying with a different server in {sleep_time} seconds...")
                time.sleep(sleep_time)
            else:
                print(f"Grid Cell {idx}: Max retries reached. Skipping.")
    return buildings


def optimize_data_types(gdf):
    """
    Optimize data types of a GeoDataFrame to save memory.
    Converts numerical columns to float32 and categorical columns to category.
    """
    # Convert numeric columns to float32
    float_cols = ['Value_USD', 'Latitude', 'Longitude', 'Distance_km', 'Damage_%', 'Loss_USD']
    for col in float_cols:
        if col in gdf.columns:
            gdf[col] = gdf[col].astype('float32')

    # Convert object columns to category if they have limited unique values
    category_cols = ['Building_Type']
    for col in category_cols:
        if col in gdf.columns:
            gdf[col] = gdf[col].astype('category')

    return gdf


def sanitize_columns(gdf):
    """
    Cleans and shortens column names to avoid conflicts with GeoPackage.
    """
    gdf.columns = [col[:63] if len(col) > 63 else col for col in gdf.columns]  # Shorten long names
    gdf.columns = [col.replace(":", "_").replace("-", "_").replace(" ", "_") for col in gdf.columns]  # Sanitize names
    gdf = gdf.loc[:, ~gdf.columns.duplicated()]  # Remove duplicate columns
    return gdf

def save_to_geopackage(gdf, filename, layer="buildings"):
    """
    Saves a GeoDataFrame to a GeoPackage, ensuring compatibility.
    """
    try:
        gdf = sanitize_columns(gdf)
        gdf.to_file(filename, driver='GPKG', layer=layer)
        print(f"GeoDataFrame saved to {filename}")
    except Exception as e:
        print(f"Error saving GeoDataFrame to GeoPackage: {e}")



def get_osm_buildings(place_name="Texas, USA", grid_size_km=50, 
                      max_retries=3, sleep_time=5, parallel=True, max_workers=2, batch_size=10):
    """
    Fetches building data from OSM for the specified place (Texas, USA).
    Caches data in a GeoPackage for faster repeated runs.
    Processes data in batches to manage memory usage.
    """
    logging.basicConfig(filename='osm_fetching.log', level=logging.WARNING,
                        format='%(asctime)s %(levelname)s:%(message)s')

    cache_filename = 'texas_buildings.gpkg'

    if os.path.exists(cache_filename):
        print("Loading cached building data (GeoPackage)...")
        combined_buildings = gpd.read_file(cache_filename)
        print(f"Total buildings loaded from cache: {len(combined_buildings)}")
        return combined_buildings

    # 1. Get boundary of Texas
    try:
        print(f"Fetching boundary for {place_name}...")
        gdf_place = ox.geocode_to_gdf(place_name)
        print("Boundary fetched.")
    except Exception as e:
        print(f"Error fetching boundary: {e}")
        return gpd.GeoDataFrame()

    # 2. Create grid
    print(f"Splitting '{place_name}' into {grid_size_km} km grid cells...")
    grid_clipped = split_polygon_into_grid(gdf_place, grid_size_km=grid_size_km)
    print(f"Total grids created: {len(grid_clipped)}")

    # 3. Parallel fetch in batches
    all_buildings = []
    total_grids = len(grid_clipped)
    print(f"Total grids to process: {total_grids}")

    for start in range(0, total_grids, batch_size):
        end = min(start + batch_size, total_grids)
        batch = grid_clipped.iloc[start:end]
        print(f"Processing batch {start // batch_size + 1} ({start} to {end - 1})")

        if parallel:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_idx = {
                    executor.submit(_fetch_buildings_for_polygon, row.geometry, idx, max_retries, sleep_time): idx
                    for idx, row in batch.iterrows()
                }
                for future in concurrent.futures.as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        buildings = future.result()
                        if not buildings.empty:
                            buildings["Grid_Cell_ID"] = idx
                            all_buildings.append(buildings)
                    except Exception as e:
                        print(f"Grid Cell {idx} exception: {e}")
        else:
            for idx, row in batch.iterrows():
                buildings = _fetch_buildings_for_polygon(row.geometry, idx, max_retries, sleep_time)
                if not buildings.empty:
                    buildings["Grid_Cell_ID"] = idx
                    all_buildings.append(buildings)

        # Optional: Save intermediate results to disk to free memory
        if len(all_buildings) > 0:
            combined_batch = pd.concat(all_buildings, ignore_index=True)
            combined_batch = gpd.GeoDataFrame(combined_batch, crs="EPSG:4326")
            batch_num = start // batch_size + 1
            batch_filename = f"batch_{idx}.gpkg"
            save_to_geopackage(combined_batch, batch_filename, layer='buildings')
            # batch_filename = f'texas_buildings_batch_{batch_num}.gpkg'
            # combined_batch.to_file(batch_filename, driver='GPKG', encoding='UTF-8', layer='buildings')
            print(f"Batch {batch_num} saved as '{batch_filename}'.")
            all_buildings = []  # Clear list to free memory
            gc.collect()  # Force garbage collection

    # 4. Combine all batch files
    combined_buildings = gpd.GeoDataFrame()
    for start in range(0, total_grids, batch_size):
        batch_num = start // batch_size + 1
        batch_file = f'texas_buildings_batch_{batch_num}.gpkg'
        if os.path.exists(batch_file):
            batch_gdf = gpd.read_file(batch_file)
            combined_buildings = pd.concat([combined_buildings, batch_gdf], ignore_index=True)
            print(f"Batch {batch_num} loaded from '{batch_file}'.")
            os.remove(batch_file)  # Delete batch file after combining to save disk space
            print(f"Batch file '{batch_file}' removed.")
    
    if not combined_buildings.empty:
        combined_buildings = gpd.GeoDataFrame(combined_buildings, crs="EPSG:4326")
        print(f"\nTotal buildings fetched: {len(combined_buildings)}")

        # Optimize data types to save memory
        combined_buildings = optimize_data_types(combined_buildings)

        # Post-processing
        combined_buildings.reset_index(drop=True, inplace=True)

        # Unique ID
        if 'osmid' in combined_buildings.columns:
            combined_buildings.rename(columns={'osmid': 'Building_ID'}, inplace=True)
        else:
            combined_buildings['Building_ID'] = [str(uuid.uuid4()) for _ in range(len(combined_buildings))]

        if 'building' in combined_buildings.columns:
            combined_buildings.rename(columns={'building': 'Building_Type'}, inplace=True)
        else:
            combined_buildings['Building_Type'] = 'other'

        combined_buildings.dropna(subset=['geometry'], inplace=True)

        # Synthetic asset values
        value_ranges = {
            'residential': (100000, 500000),
            'commercial': (200000, 1000000),
            'industrial': (300000, 1500000),
            'retail': (150000, 800000),
            'office': (200000, 1000000),
            'other': (100000, 700000)
        }
        def assign_asset_value(btype):
            btype_l = str(btype).lower()
            val_range = value_ranges.get(btype_l, value_ranges['other'])
            return np.random.uniform(val_range[0], val_range[1])

        combined_buildings['Value_USD'] = combined_buildings['Building_Type'].apply(assign_asset_value).astype('float32')

        # Centroids (use EPSG:32614 for Texas region)
        combined_proj = combined_buildings.to_crs(epsg=32614)
        combined_proj['centroid'] = combined_proj.geometry.centroid
        centroids_geo = combined_proj.set_geometry('centroid').to_crs(epsg=4326)
        combined_buildings['Latitude'] = centroids_geo['centroid'].y.astype('float32')
        combined_buildings['Longitude'] = centroids_geo['centroid'].x.astype('float32')

        # Save to cache
        combined_buildings.to_file(cache_filename, driver='GPKG', encoding='UTF-8', layer='buildings', compress=True)
        print(f"Building data saved to '{cache_filename}'.")
    else:
        combined_buildings = gpd.GeoDataFrame(
            columns=['Building_ID', 'Building_Type', 'geometry', 'Latitude', 'Longitude', 'Value_USD']
        )
        print("No buildings fetched.")

    # Final garbage collection
    gc.collect()
    return combined_buildings


ERA5_FILEPATH = 'era5_Harvey_aug23-31_2017_extended.nc'  # Update if necessary

LANDFALL_DETECTION_METHOD = 'max_wind'

def open_and_prepare_dataset(filepath=ERA5_FILEPATH):
    """
    Opens the ERA5 NetCDF file, renames valid_time->time if needed,
    calculates wind_speed_kph, wind_gust_kph, and msl_hPa if they exist.
    Returns an xarray.Dataset
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"ERA5 file '{filepath}' not found.")
    
    ds = xr.open_dataset(filepath, chunks={'time': 10})
    
    # If the dataset uses 'valid_time' instead of 'time', rename it.
    if 'valid_time' in ds.coords:
        ds = ds.rename({'valid_time': 'time'})
    
    # Ensure we have 'u10' and 'v10'
    if 'u10' not in ds.data_vars or 'v10' not in ds.data_vars:
        raise ValueError("Missing 'u10' or 'v10' in the dataset. Adjust variable names if needed.")

    # Calculate wind speed (m/s -> kph)
    ds['wind_speed_mps'] = np.sqrt(ds['u10']**2 + ds['v10']**2)
    ds['wind_speed_kph'] = ds['wind_speed_mps'] * 3.6

    # Attempt wind gust
    if 'i10fg' in ds.data_vars:
        ds['wind_gust_kph'] = ds['i10fg'] * 3.6
    elif '10si' in ds.data_vars:
        ds['wind_gust_kph'] = ds['10si'] * 3.6
    else:
        ds['wind_gust_kph'] = np.nan
    
    # Attempt MSL in hPa
    if 'msl' in ds.data_vars:
        ds['msl_hPa'] = ds['msl'] / 100.0
    else:
        ds['msl_hPa'] = np.nan
    
    return ds

def identify_landfall(df, method='max_wind'):
    """
    Identify the landfall event by grouping all grid points by time.
    method='max_wind' => picks the highest wind speed in each time step,
                         then picks the overall max across time.
    method='min_mslp' => picks the lowest MSLP in each time step,
                         then picks the overall min across time.

    Returns a single-row DataFrame with landfall details.
    """
    if 'time' not in df.columns:
        raise KeyError("No 'time' column found in DataFrame. Check the rename steps!")
    
    # Group by 'time'
    grouped = df.groupby('time', group_keys=True)
    
    if method == 'max_wind':
        time_maxwind = grouped.apply(lambda g: g.loc[g['wind_speed_kph'].idxmax()])
        landfall_event = time_maxwind.loc[time_maxwind['wind_speed_kph'].idxmax()]
    elif method == 'min_mslp':
        time_minmslp = grouped.apply(lambda g: g.loc[g['msl_hPa'].idxmin()])
        landfall_event = time_minmslp.loc[time_minmslp['msl_hPa'].idxmin()]
    else:
        raise ValueError("method must be 'max_wind' or 'min_mslp'")
    
    # Convert to a single-row DataFrame
    landfall_df = pd.DataFrame([{
        'Cyclone_ID': 1,
        'Method': method,
        'Wind_Speed_kph': landfall_event['wind_speed_kph'],
        'Wind_Gust_kph': landfall_event['wind_gust_kph'],
        'msl_hPa': landfall_event['msl_hPa'],
        'Latitude': landfall_event['latitude'],
        'Longitude': landfall_event['longitude'],
        'Time': landfall_event['time']
    }])
    
    return landfall_df

def plot_cyclone_features_at_landfall_point(df, landfall_df):
    """
    Plots wind speed, wind gust, and MSLP over time for the specific grid point
    where landfall occurs, highlighting the landfall event.
    """
    sns.set(style="whitegrid")
    plt.figure(figsize=(16, 12))

    # Extract the landfall location (lat, lon)
    landfall_lat = landfall_df.iloc[0]['Latitude']
    landfall_lon = landfall_df.iloc[0]['Longitude']

    # Filter the data for the landfall grid point
    landfall_grid_df = df[(df['latitude'] == landfall_lat) & (df['longitude'] == landfall_lon)]

    if landfall_grid_df.empty:
        print("No data available for the grid point where landfall occurs.")
        return

    # Sort by time for consistent plotting
    landfall_grid_df = landfall_grid_df.sort_values(by='time')

    # 1. Wind Speed
    plt.subplot(3, 1, 1)
    sns.lineplot(
        data=landfall_grid_df,
        x='time',
        y='wind_speed_kph',
        color='blue',
        label='Wind Speed (kph)'
    )
    plt.scatter(
        landfall_df['Time'],
        landfall_df['Wind_Speed_kph'],
        color='red',
        zorder=5,
        label='Landfall'
    )
    plt.title("Wind Speed at Landfall Grid Point Over Time", fontsize=14)
    plt.xlabel("Time")
    plt.ylabel("Wind Speed (kph)")
    plt.legend()

    # 2. Wind Gust
    plt.subplot(3, 1, 2)
    if 'wind_gust_kph' in landfall_grid_df.columns and landfall_grid_df['wind_gust_kph'].notna().any():
        sns.lineplot(
            data=landfall_grid_df,
            x='time',
            y='wind_gust_kph',
            color='orange',
            label='Wind Gust (kph)'
        )
        plt.scatter(
            landfall_df['Time'],
            landfall_df['Wind_Gust_kph'],
            color='red',
            zorder=5,
            label='Landfall'
        )
        plt.title("Wind Gust at Landfall Grid Point Over Time", fontsize=14)
        plt.xlabel("Time")
        plt.ylabel("Wind Gust (kph)")
        plt.legend()
    else:
        plt.text(0.5, 0.5, "No Wind Gust Data Available", ha='center', va='center', fontsize=14)
        plt.axis('off')

    # 3. MSLP
    plt.subplot(3, 1, 3)
    if 'msl_hPa' in landfall_grid_df.columns and landfall_grid_df['msl_hPa'].notna().any():
        sns.lineplot(
            data=landfall_grid_df,
            x='time',
            y='msl_hPa',
            color='green',
            label='MSLP (hPa)'
        )
        plt.scatter(
            landfall_df['Time'],
            landfall_df['msl_hPa'],
            color='red',
            zorder=5,
            label='Landfall'
        )
        plt.title("MSLP at Landfall Grid Point Over Time", fontsize=14)
        plt.xlabel("Time")
        plt.ylabel("MSLP (hPa)")
        plt.legend()
    else:
        plt.text(0.5, 0.5, "No MSLP Data Available", ha='center', va='center', fontsize=14)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('Harvey_landfall_grid_point.png', bbox_inches='tight')
    plt.show()

def print_cyclone_summary(landfall_df):
    """
    Prints a summary of the cyclone at landfall, including key features.
    """
    row = landfall_df.iloc[0]
    print("\n--- Landfall Summary ---")
    print(f"Method of Detection  : {row['Method']}")
    print(f"Wind Speed (kph)     : {row['Wind_Speed_kph']:.1f}")
    print(f"Wind Gust (kph)      : {row['Wind_Gust_kph']:.1f}")
    print(f"MSL (hPa)            : {row['msl_hPa']:.1f}")
    print(f"Latitude, Longitude  : {row['Latitude']:.2f}, {row['Longitude']:.2f}")
    print(f"Time                 : {row['Time']}")
    print("------------------------")

def calculate_damage_percentage_kph(wind_speed_kph, distance_km, radius_km, building_type):
    """
    Basic wind-speed damage function:
    160 kph -> 0%, 320 kph -> 100%, decreasing with distance.
    Adjust as needed for U.S. hurricanes.
    """
    if distance_km > radius_km:
        return 0

    damage_at_center = (wind_speed_kph - 160) / 160 * 100
    damage_at_center = np.clip(damage_at_center, 0, 100)

    building_type_lower = building_type.lower() if isinstance(building_type, str) else 'other'
    resilience_factors = {
        'residential': 0.8,
        'commercial': 0.6,
        'industrial': 0.5,
        'retail': 0.6,
        'office': 0.6,
        'other': 0.7
    }
    resilience = resilience_factors.get(building_type_lower, 0.7)

    adjusted_damage = damage_at_center * (1 - resilience)
    damage = adjusted_damage * (1 - distance_km / radius_km)
    return np.clip(damage, 0, 100)


def calculate_losses(assets_gdf, cyclone_df, radius_km=160):
    """
    Calculates financial losses for assets from Hurricane Harvey 
    based on wind speed & distance.
    """
    loss_records = []

    for _, cyclone in cyclone_df.iterrows():
        cyclone_id = cyclone['Cyclone_ID']
        wind_speed = cyclone['Wind_Speed_kph']
        lat = cyclone['Latitude']
        lon = cyclone['Longitude']
        center_point = Point(lon, lat)

        # Project to UTM for accurate distance calculations
        assets_gdf_proj = assets_gdf.to_crs(epsg=32614)
        center_point_proj = gpd.GeoSeries([center_point], crs="EPSG:4326").to_crs(epsg=32614).iloc[0]

        # Distance in km
        assets_gdf_proj['Distance_km'] = assets_gdf_proj.geometry.distance(center_point_proj) / 1000.0

        # Damage calculation
        assets_gdf_proj['Damage_%'] = assets_gdf_proj.apply(
            lambda row: calculate_damage_percentage_kph(
                wind_speed_kph=wind_speed,
                distance_km=row['Distance_km'],
                radius_km=radius_km,
                building_type=row['Building_Type']
            ),
            axis=1
        ).astype('float32')

        # Financial loss
        assets_gdf_proj['Loss_USD'] = (assets_gdf_proj['Value_USD'] * (assets_gdf_proj['Damage_%'] / 100.0)).astype('float32')

        # Total loss
        total_loss = assets_gdf_proj['Loss_USD'].sum()
        loss_records.append({
            'Cyclone_ID': cyclone_id,
            'Wind_Speed_kph': wind_speed,
            'Latitude': lat,
            'Longitude': lon,
            'Total_Loss_USD': total_loss
        })

        # Merge back
        assets_gdf['Distance_km'] = assets_gdf_proj['Distance_km']
        assets_gdf['Damage_%'] = assets_gdf_proj['Damage_%']
        assets_gdf['Loss_USD'] = assets_gdf_proj['Loss_USD']

        # Clean up
        del assets_gdf_proj
        gc.collect()

    loss_df = pd.DataFrame(loss_records)
    return loss_df, assets_gdf


def create_loss_summary_plot(loss_df):
    """
    Creates a bar plot summarizing total financial losses.
    """
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 6))
    sns.barplot(x='Cyclone_ID', y='Total_Loss_USD', data=loss_df, palette='Reds_d')
    plt.title('Total Financial Loss - Hurricane Harvey')
    plt.xlabel('Cyclone ID')
    plt.ylabel('Total Loss (USD)')
    plt.tight_layout()
    plt.show()


def create_interactive_map(assets_gdf, cyclone_df, radius_km=160):
    """
    Creates an interactive Folium map showing asset losses and cyclone impact.
    """
    for _, cyclone in cyclone_df.iterrows():
        lat = cyclone['Latitude']
        lon = cyclone['Longitude']
        wind_speed = cyclone['Wind_Speed_kph']
        cyclone_id = cyclone['Cyclone_ID']
        total_loss = cyclone['Total_Loss_USD']

        m = folium.Map(location=[lat, lon], zoom_start=6)

        # Impact circle
        folium.Circle(
            location=[lat, lon],
            radius=radius_km * 1000,
            color='blue',
            fill=True,
            fill_opacity=0.2,
            popup=f"Hurricane ID: {cyclone_id}, Wind: {wind_speed:.1f} kph"
        ).add_to(m)

        # Center marker
        folium.Marker(
            location=[lat, lon],
            icon=folium.Icon(color='blue', icon='info-sign'),
            popup=f"Hurricane Harvey Center"
        ).add_to(m)

        # Add asset markers
        for _, asset in assets_gdf.iterrows():
            if 'Loss_USD' not in asset:
                continue  # Skip if loss data is not available

            color = 'red' if asset['Loss_USD'] > 0 else 'green'
            folium.CircleMarker(
                location=[asset['Latitude'], asset['Longitude']],
                radius=3,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.6,
                popup=(
                    f"Building ID: {asset['Building_ID']}<br>"
                    f"Type: {asset['Building_Type']}<br>"
                    f"Value: ${asset['Value_USD']:,.2f}<br>"
                    f"Damage: {asset['Damage_%']:.1f}%<br>"
                    f"Loss: ${asset['Loss_USD']:,.2f}"
                )
            ).add_to(m)

        # Add legend
        legend_html = '''
         <div style="position: fixed; 
                     bottom: 50px; left: 50px; width: 150px; height: 90px; 
                     border:2px solid grey; z-index:9999; font-size:14px;
                     background-color:white;
                     ">
         &nbsp;<b>Legend</b><br>
         &nbsp;<i class="fa fa-circle" style="color:red"></i>&nbsp;Loss<br>
         &nbsp;<i class="fa fa-circle" style="color:green"></i>&nbsp;No Loss
         </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))

        # Save map to HTML file
        map_filename = f"hurricane_{cyclone_id}_map.html"
        m.save(map_filename)
        print(f"Interactive map saved as '{map_filename}'.")

    # Force garbage collection after map creation
    gc.collect()


# -------------------------------
# 3. Main Execution Function
# -------------------------------

def main():
    # 1. Fetch OSM Building Data for Texas
    print("Fetching OSM building data for Texas...")
    assets_gdf = get_osm_buildings(
        place_name="Texas, USA",
        grid_size_km=100,
        parallel=True,
        max_workers=6
    )
    print(f"Total Buildings Fetched: {len(assets_gdf)}")
    if assets_gdf.empty:
        print("No building data. Exiting.")
        return

    print("\nSample Buildings:")
    print(assets_gdf.head())

    # 2. Process ERA5 Data for Cyclone Analysis
    print("\nProcessing ERA5 data for Hurricane Harvey...")
    ds = open_and_prepare_dataset(ERA5_FILEPATH)
    df = ds[['wind_speed_kph', 'wind_gust_kph', 'msl_hPa', 'latitude', 'longitude', 'time']].to_dataframe().reset_index()
    landfall_df = identify_landfall(df, method=LANDFALL_DETECTION_METHOD)
    print_cyclone_summary(landfall_df)

    # 3. Plot Cyclone Features at Landfall Grid Point
    plot_cyclone_features_at_landfall_point(df, landfall_df)

    # 4. Calculate Losses
    print("\nCalculating losses based on Hurricane Harvey impact...")
    loss_df, updated_assets_gdf = calculate_losses(assets_gdf, landfall_df, radius_km=160)
    print("\nLoss Summary:")
    print(loss_df)

    # 5. Plot Loss Summary
    if not loss_df.empty:
        create_loss_summary_plot(loss_df)
    else:
        print("No loss data to plot.")

    # 6. Interactive Map
    if not updated_assets_gdf.empty and not landfall_df.empty:
        create_interactive_map(updated_assets_gdf, landfall_df, radius_km=160)
    else:
        print("Insufficient data for interactive map.")

# Entry Point
if __name__ == "__main__":
    main()
