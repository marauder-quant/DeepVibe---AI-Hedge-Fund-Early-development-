# Task 31: Fix QMAC CSV Export to Provide Complete Metrics for All Parameters

## Description
Fix the QMAC strategy CSV export functionality to ensure all parameter combinations have their complete performance metrics properly exported, not just the top-ranked combinations.

## Background
The QMAC (Quad Moving Average Crossover) strategy exports backtest results to CSV files. Currently, only the top-ranked parameter combination for each timeframe has complete metrics (total_return, sharpe_ratio, max_drawdown, num_trades, win_rate), while all other combinations only have the "performance" metric filled in with missing values for the other metrics. This limitation makes it difficult to properly compare and analyze all tested parameter combinations.

## Objectives
1. Modify the QMAC database and export functions to store complete metrics for all parameter combinations
2. Ensure all exported parameter combinations in the CSV have their full set of performance metrics populated
3. Maintain backward compatibility with existing database structure
4. Update the export process to handle the enhanced data format
5. Test the changes to verify all metrics are properly exported

## Instructions for AI Agent

### Step 1: Understand the Current Export Process
- Review the current export functionality in `backtests/qmac_strategy/src/export_qmac_db.py`
- Analyze how parameters are saved to the database in `save_top_parameters_to_db` function
- Understand how the `get_parameters_from_db` function retrieves and formats the data
- Identify why only the top-ranked parameter combination has complete metrics

### Step 2: Modify Parameter Storage Logic
- Update the `save_top_parameters_to_db` function in `lib/database.py` to store complete metrics for all parameter combinations
- Ensure performance statistics (total_return, sharpe_ratio, max_drawdown, num_trades, win_rate) are calculated and saved for each parameter set
- Modify the code to run a full backtest for each parameter combination to gather complete metrics
- Consider adding a parameter to control the depth of metrics calculation (for performance reasons)

### Step 3: Enhance the Export Function
- Update the `export_qmac_db_to_csv` function to ensure all metrics are included in the export
- Verify that the DataFrame structure maintains all performance metrics
- Ensure the CSV export includes all columns with appropriate formatting
- Add progress reporting for long-running exports with many parameter combinations

### Step 4: Update Related Code
- Check for any dependent code that might be affected by the changes
- Update the database query functions to handle the enhanced data model
- Make sure the `qmac_main.py` script properly runs backtests for all parameter combinations when needed

### Step 5: Testing and Validation
- Test the modified export functionality with various parameter combinations
- Verify that all rows in the exported CSV have complete metrics
- Compare results with the previous implementation to ensure consistency
- Check performance to ensure the changes don't introduce significant slowdowns

## Success Criteria
- All parameter combinations in the exported CSV file have complete metrics populated
- The export process works efficiently and doesn't introduce significant performance penalties
- Exported data is consistent with the previous implementation for the same parameter combinations
- No regression in existing functionality or database operations
- A sample CSV export shows all metrics populated for all parameter combinations

## Resources
- The QMAC database export code in `backtests/qmac_strategy/src/export_qmac_db.py`
- The database utility functions in `backtests/qmac_strategy/src/lib/database.py`
- The QMAC strategy core implementations in `backtests/qmac_strategy/src/lib/strategy_core.py`
- The parameter optimization code in `backtests/qmac_strategy/src/lib/optimizer.py` 