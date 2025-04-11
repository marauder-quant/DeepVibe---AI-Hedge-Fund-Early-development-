#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Quad Moving Average Crossover (QMAC) strategy implementation using vectorbt and Alpaca data.
This module provides the original implementation, now reimplemented in a modular way.
See qmac_main.py for the entry point to the refactored version.
"""

import os
import sys

print("This file has been refactored into modular components.")
print("Please use qmac_main.py as the entry point.")
print("For more information, see the README.md file.")

# Redirect to the new entry point
if __name__ == "__main__":
    import argparse
    import sys
    
    # Get the arguments passed to this script
    parser = argparse.ArgumentParser(description='QMAC Strategy (Legacy)')
    args, unknown = parser.parse_known_args()
    
    # Forward to the new script
    from qmac_main import main
    sys.argv[0] = 'qmac_main.py'  # Replace script name
    main()