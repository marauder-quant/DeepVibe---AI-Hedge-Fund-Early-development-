#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fast QMAC strategy runner.
This is an optimized script for running the QMAC strategy with high performance.
"""

import os
import sys
import time
from datetime import datetime
import argparse

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.qmac_main import main as qmac_main

if __name__ == "__main__":
    # Parse any command line arguments
    parser = argparse.ArgumentParser(description="High-performance QMAC backtester")
    parser.add_argument("--symbol", default="SPY", help="Trading symbol (default: SPY)")
    parser.add_argument("--fast", action="store_true", help="Run with limited window range for quick testing")
    parser.add_argument("--no-ray", action="store_true", help="Disable Ray distributed computing")
    parser.add_argument("--cores", type=int, default=None, help="Number of CPU cores to use (default: auto)")
    parser.add_argument("--batch-size", type=int, default=500, help="Batch size for Ray (default: calculated dynamically)")
    
    args = parser.parse_args()
    
    # Prepare arguments to pass to qmac_main
    sys_args = [
        "--symbol", args.symbol,
        "--window-step", "5",  # Increase step size for faster processing
    ]
    
    if args.fast:
        sys_args.append("--fast")
    
    if args.no_ray:
        sys_args.extend(["--use-ray", "False"])
    
    if args.cores:
        sys_args.extend(["--num-cpus", str(args.cores)])
    
    # Set environment variables for Ray tuning
    os.environ["RAY_DISABLE_MEMORY_MONITOR"] = "1"  # Disable memory monitoring for better performance
    
    if args.batch_size:
        os.environ["QMAC_BATCH_SIZE"] = str(args.batch_size)
    
    # Save original arguments
    original_argv = sys.argv
    
    try:
        # Replace sys.argv with our custom arguments
        sys.argv = [sys.argv[0]] + sys_args
        
        print("=" * 80)
        print(f"QMAC HIGH PERFORMANCE MODE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Running with arguments: {' '.join(sys_args)}")
        print("=" * 80)
        
        start_time = time.time()
        qmac_main()
        end_time = time.time()
        
        print("\nQMAC High Performance run completed in {:.2f} seconds".format(end_time - start_time))
    
    finally:
        # Restore original argv
        sys.argv = original_argv 