import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from shapely.geometry import Point
import warnings

warnings.filterwarnings("ignore")  # Suppress warnings for cleaner output


ERA5_FILEPATH = 'era5_Harvey_aug23-31_2017_extended.nc'


LANDFALL_DETECTION_METHOD = 'max_wind' # or 'min_mslp'


def open_and_prepare_dataset(filepath=ERA5_FILEPATH):
    """
    Opens the ERA5 NetCDF file, renames valid_time->time if needed,
    calculates wind_speed_kph, wind_gust_kph, and msl_hPa if they exist.
    Returns an xarray.Dataset
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"ERA5 file '{filepath}' not found.")
    
    ds = xr.open_dataset(filepath, chunks={'time': 10})
    

    if 'valid_time' in ds.coords:
        ds = ds.rename({'valid_time': 'time'})
    

    if 'u10' not in ds.data_vars or 'v10' not in ds.data_vars:
        raise ValueError("Missing 'u10' or 'v10' in the dataset. Adjust variable names if needed.")


    ds['wind_speed_mps'] = np.sqrt(ds['u10']**2 + ds['v10']**2)
    ds['wind_speed_kph'] = ds['wind_speed_mps'] * 3.6


    if 'i10fg' in ds.data_vars:
        ds['wind_gust_kph'] = ds['i10fg'] * 3.6
    elif '10si' in ds.data_vars:
        ds['wind_gust_kph'] = ds['10si'] * 3.6
    else:
        ds['wind_gust_kph'] = np.nan
    

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
    

    grouped = df.groupby('time', group_keys=True)
    
    if method == 'max_wind':
        time_maxwind = grouped.apply(lambda g: g.loc[g['wind_speed_kph'].idxmax()])
        landfall_event = time_maxwind.loc[time_maxwind['wind_speed_kph'].idxmax()]
    elif method == 'min_mslp':
        time_minmslp = grouped.apply(lambda g: g.loc[g['msl_hPa'].idxmin()])
        landfall_event = time_minmslp.loc[time_minmslp['msl_hPa'].idxmin()]
    else:
        raise ValueError("method must be 'max_wind' or 'min_mslp'")
    

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


    landfall_lat = landfall_df.iloc[0]['Latitude']
    landfall_lon = landfall_df.iloc[0]['Longitude']


    landfall_grid_df = df[(df['latitude'] == landfall_lat) & (df['longitude'] == landfall_lon)]

    if landfall_grid_df.empty:
        print("No data available for the grid point where landfall occurs.")
        return


    landfall_grid_df = landfall_grid_df.sort_values(by='time')

    # Wind Speed
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

    # Wind Gust
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

    # MSLP
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


def main():

    ds = open_and_prepare_dataset(ERA5_FILEPATH)
    

    df = ds[['wind_speed_kph', 'wind_gust_kph', 'msl_hPa', 'latitude', 'longitude', 'time']].to_dataframe().reset_index()
    
    # Identify the landfall event
    landfall_df = identify_landfall(df, method=LANDFALL_DETECTION_METHOD)
    

    print_cyclone_summary(landfall_df)
    

    plot_cyclone_features_at_landfall_point(df, landfall_df)

if __name__ == "__main__":
    main()
