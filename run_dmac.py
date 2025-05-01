#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DMAC Strategy Runner
Easy-to-use wrapper script to run the DMAC strategy without import errors.
"""

import os
import sys
import time
from datetime import datetime

# Add the project root to Python path to fix imports
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

def main():
    # Import here after path is set up
    from backtests.dmac_strategy.dmac_main import main as dmac_main
    
    # Display header
    print("="*80)
    print(f"DMAC STRATEGY RUNNER - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Track execution time
    start_time = time.time()
    
    # Run the DMAC main function
    dmac_main()
    
    # Display execution time
    end_time = time.time()
    print(f"\nDMAC Strategy execution completed in {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main() 