# -*- coding: utf-8 -*-
# Copyright (c) 2024 Jivesh Dixit, P.S. II, NCMRWF
# All rights reserved.
#
# This software is licensed under MIT license.
# Contact [jdixit@nic.in; jiveshdixit@gmail.com]



###### catastrophe_model/__init__.py ######

from plotting import (
    plot_cyclone_features_at_landfall_point,
    create_loss_summary_plot,
    create_interactive_map
)

from hazard import (
    open_and_prepare_dataset,
    identify_landfall
)

from utils import (
    split_polygon_into_grid,
    _fetch_buildings_for_polygon,
    optimize_data_types,
    sanitize_columns,
    save_to_geopackage,
    print_cyclone_summary,
    download_era5_data,
    write_loss_data_to_csv
)

from exposure import (
    calculate_damage_percentage,
    calculate_damage_from_precipitation,
    calculate_losses
)

from vulnerability import (
    get_osm_buildings
)

from workflow import (
    create_interactive_map_workflow
)
