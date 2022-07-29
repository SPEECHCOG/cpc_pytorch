#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Einari Vaaras, einari.vaaras@tuni.fi, Tampere University
Speech and Cognition Research Group, https://webpages.tuni.fi/specog/index.html

"""

def convert_py_conf_file_to_text(conf_file_name):
    """
    Read the lines of a .py configuration file into a list, skip the lines with comments in them.
    _____________________________________________________________________________________________
    Input parameters:
        
    conf_file_name: The name of the configuration file we want to read.
    
    _____________________________________________________________________________________________
    """
    
    lines = []
    multiline_comment = 0
    with open(conf_file_name) as f:
        for line in f:
            if len(line.rstrip()) > 0:
                if line.rstrip()[0] != '#':
                    if not multiline_comment:
                        if len(line.rstrip()) > 2:
                            if line.rstrip()[0:3] == '"""' or line.rstrip()[0:3] == "'''":
                                if line.rstrip().count('"""') == 1 or line.rstrip().count("'''") == 1:
                                    multiline_comment = 1
                            else:
                                lines.append(line.rstrip())
                        else:
                            lines.append(line.rstrip())
                    else:
                        if len(line.rstrip()) > 2:
                            if line.rstrip()[0:3] == '"""' or line.rstrip()[0:3] == "'''":
                                if line.rstrip().count('"""') == 1 or line.rstrip().count("'''") == 1:
                                    multiline_comment = 0
            
    return lines
