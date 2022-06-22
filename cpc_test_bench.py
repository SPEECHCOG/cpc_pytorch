# -*- coding: utf-8 -*-
"""
Author: Einari Vaaras, einari.vaaras@tuni.fi, Tampere University
Speech and Cognition Research Group, https://webpages.tuni.fi/specog/index.html

Code for training and evaluating a number of different CPC models.

"""

import sys
import os


# Read the configuration file
if len(sys.argv) != 2:
    sys.exit('''Usage:\n  python cpc_test_bench.py <dir_to_configuration_files>\n\n
             where <dir_to_configuration_files> is a directory containing at least one .py configuration file.''')
else:
    dir_of_conf_files = sys.argv[1]
    
# Check if the given directory is actually a directory
if not os.path.isdir(dir_of_conf_files):
    sys.exit(f'The given argument {dir_of_conf_files} is not a directory!')

# Find out the configuration files in the given directory
filenames_in_dir = os.listdir(dir_of_conf_files)

# Clean the list if there are other files than .py configuration files
conf_file_names = [filename for filename in filenames_in_dir if filename.endswith('.py')]

# Go through each configuration file and run train_cpc_model.py using the given settings
conf_file_index = 1
for conf_file in conf_file_names:
    os.system(f'echo -n "Running CPC experiment {conf_file_index}/{len(conf_file_names)} "')
    os.system(f'echo "using configuration file {conf_file} (see log file of {conf_file} for further details)"')
    os.system(f'python train_cpc_model.py {os.path.join(dir_of_conf_files, conf_file)}')
    conf_file_index += 1
