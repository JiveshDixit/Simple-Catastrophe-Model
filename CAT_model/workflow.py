# -*- coding: utf-8 -*-
# Copyright (c) 2024 Jivesh Dixit, P.S. II, NCMRWF
# All rights reserved.
#
# This software is licensed under MIT license.
# Contact [jdixit@nic.in; jiveshdixit@gmail.com]

###### catastrophe_model/workflow.py ######

import logging
import pandas as pd
import geopandas as gpd
from hazard import open_and_prepare_dataset, identify_landfall
from vulnerability import get_osm_buildings
from exposure import calculate_losses
from plotting import create_interactive_map
from utils import print_cyclone_summary, write_loss_data_to_csv
from plotting import (plot_cyclone_features_at_landfall_point, create_loss_summary_plot)#, create_detailed_loss_summary_plot

def create_interactive_map_workflow(
    cyclone_name: str,
    north: float,
    south: float,
    east: float,
    west: float,
    start_date: str,
    end_date: str,
    place_name: str,
    grid_size_km: int,
    parallel: bool,
    max_workers: int,
    batch_size: int,
    radius_km: float
):
    """
    Orchestrates the entire workflow to create an interactive loss map of cyclone impact:
      . Download and process ERA5 data
      . Identify landfall
      . Fetch OSM building data
      . Calculate losses
      . Create interactive map

    """
    logging.info("===== Starting Interactive Map Workflow =====")

    # Download & process ERA5 data
    logging.info(f"Processing ERA5 data for {cyclone_name}...")
    ds = open_and_prepare_dataset(
        cyclone_name=cyclone_name,
        north=north,
        south=south,
        east=east,
        west=west,
        start_date=start_date,
        end_date=end_date
    )
    df = ds.to_dataframe().reset_index()

    # Identify landfall
    logging.info("Identifying landfall...")
    landfall_df = identify_landfall(df, method='max_wind')
    print_cyclone_summary(landfall_df)

    plot_cyclone_features_at_landfall_point(df, landfall_df, cyclone_name)

    # Fetch OSM building data
    logging.info(f"Fetching OSM building data for {place_name}...")
    buildings_gdf = get_osm_buildings(
        place_name=place_name,
        grid_size_km=grid_size_km,
        parallel=parallel,
        max_workers=max_workers,
        batch_size=batch_size
    )
    logging.info(f"Total Buildings Fetched: {len(buildings_gdf)}")

    if buildings_gdf.empty:
        logging.warning("No building data available. Exiting workflow.")
        return

    # Calculate losses
    logging.info("Calculating losses...")
    loss_df, updated_assets_gdf = calculate_losses(buildings_gdf, landfall_df, radius_km=radius_km)
    # print(loss_df.columns, loss_df)
    # print(updated_assets_gdf.columns, updated_assets_gdf)
    # print
    if loss_df.empty:
        logging.warning("No losses were calculated (perhaps no buildings in the impact area).")
    else:
    # if not loss_df.empty:
        write_loss_data_to_csv(updated_assets_gdf, output_filename=f"{cyclone_name}_losses.csv")
        logging.info(f"Loss summary:\n{loss_df}")

    create_loss_summary_plot(loss_df, cyclone_name)
    # create_detailed_loss_summary_plot(loss_df, cyclone_name)

    # Create the interactive map
    logging.info("Creating the interactive Folium map...")
    if not loss_df.empty and "Cyclone_ID" in landfall_df.columns:
        # Optionally merge total loss into landfall_df so the map can display it
        loss_df_unique = loss_df.drop_duplicates(subset=['Cyclone_ID'])
        landfall_df = pd.merge(
            landfall_df,
            loss_df_unique[['Cyclone_ID', 'Total_Loss_USD', 'Total_Precip_mm']],
            on='Cyclone_ID',
            how='left'
        )



    create_interactive_map(cyclone_name,
        assets_gdf=updated_assets_gdf,
        cyclone_df=landfall_df,
        radius_km=radius_km
    )

    logging.info("===== Interactive Map Workflow Completed =====")
