# Task 28: Improve Out-of-Sample Backtest Result Labeling and Naming

## Description
Enhance the labeling and file naming convention of QMAC strategy out-of-sample backtest results to clearly identify stock symbols, timeframes, and sample windows used in testing.

## Background
Currently, backtest result files are only labeled with timestamps (e.g., `qmac_oos_results_20250411_005755.csv`), making it difficult to identify which stock, timeframe, and sample window a particular test used. This lack of clear labeling creates challenges when analyzing multiple test results and comparing different backtests.

## Objectives
1. Update the output file naming convention to include stock symbols and timeframes
2. Clearly label sample window information in output files
3. Improve visualization titles to indicate test parameters
4. Create consistent naming across all output file types (CSV, JSON, PNG)
5. Ensure backward compatibility with existing code that uses these files

## Instructions for AI Agent

### Step 1: Analyze Current Output Files
- Review the current naming structure of output files in the `backtests/qmac_strategy/results` directory
- Identify the core components that need to be included in the improved naming scheme
- Map the relationship between parameters in the backtest functions and the output files
- Document all file types that need to be updated (CSV, JSON, PNG, etc.)

### Step 2: Design New Naming Convention
- Create a naming convention that includes:
  - Stock symbol
  - Timeframe (e.g., 30m, 1d)
  - Sample window dates or ranges (in a compact format)
  - Strategy parameters identifier (if applicable)
  - Timestamp (for uniqueness)
- Ensure the naming is consistent across all file types
- Design a format that remains reasonably concise while providing all necessary information
- Example: `qmac_oos_GEN_30m_20180107-20250110_20250411_005755.csv`

### Step 3: Update Results File Generation Code
- Modify the code that generates output files to implement the new naming convention
- Ensure output CSV files include clear headers that identify stock, timeframe, and sample window
- Update JSON summary files to include metadata about the test parameters
- Add proper titles and labels to visualization charts
- Implement functions to parse the new filenames for backward compatibility

### Step 4: Update Visualization Labels
- Enhance chart titles to include stock symbol, timeframe, and date ranges
- Add parameter information to confidence report charts
- Ensure all axes are clearly labeled with units and descriptions
- Include summary statistics in chart legends or captions where appropriate

### Step 5: Test Changes
- Run sample backtests to verify the new naming convention works correctly
- Check that all file types are generated with consistent naming
- Verify that visualization titles and labels display correctly
- Ensure existing code can still access and process the files correctly

## Success Criteria
- All output files follow the new naming convention that clearly identifies stock, timeframe, and sample window
- Visualization titles and labels show relevant test parameters
- CSV files have clear headers indicating what data they contain
- JSON files include comprehensive metadata about the test
- Existing code continues to function with the new file naming scheme
- Users can easily identify and compare results from different backtest runs

## Resources
- Review existing file generation code in `backtests/qmac_strategy/walk_forward_optimization/`
- Examine the output files in `backtests/qmac_strategy/results/`
- Check visualization generation in `backtests/qmac_strategy/src/`
- Study how `confidence_tracker.json` stores information about multiple tests
