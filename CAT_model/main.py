# -*- coding: utf-8 -*-
# Copyright (c) 2024 Jivesh Dixit, P.S. II, NCMRWF
# All rights reserved.
#
# This software is licensed under MIT license.
# Contact [jdixit@nic.in; jiveshdixit@gmail.com]



###### catastrophe_model/main.py ######

import logging
from workflow import create_interactive_map_workflow

def main():

    logging.basicConfig(
        filename='catastrophe_model.log',
        level=logging.INFO,
        format='%(asctime)s %(levelname)s:%(message)s'
    )


    create_interactive_map_workflow(
        cyclone_name="Haiyan",
        north = 20.0,
        south = 5.0,
        east = 130.0,
        west = 115.0,
        start_date="2013-11-06",
        end_date="2013-11-11",
        place_name="Philippines",
        grid_size_km=100,
        parallel=True,
        max_workers=20,
        batch_size=10,
        radius_km=160
    )

if __name__ == "__main__":
    main()
