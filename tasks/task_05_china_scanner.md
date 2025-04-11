# Task 5: Working China Scanner and China Allocation

## Description
Develop a functional China stock scanner and implement allocation logic for Chinese securities in the portfolio.

## Background
The trading system needs to expand its capabilities to include analysis and trading of Chinese securities. This requires implementing a scanner to identify promising Chinese stocks and developing allocation logic to determine appropriate position sizes within the portfolio.

## Objectives
1. Create a specialized scanner for Chinese stocks
2. Implement allocation logic for Chinese securities
3. Ensure proper integration with the existing trading system
4. Copy the implementation into a new directory as specified in the task

## Instructions for AI Agent

### Step 1: Research and Preparation
- Identify available data sources for Chinese securities (exchanges, ADRs, ETFs)
- Determine what metrics and criteria would be most relevant for scanning Chinese stocks
- Review existing scanner implementations in the codebase to understand the architecture
- Research specific considerations for Chinese market trading (timing, regulations, etc.)

### Step 2: Scanner Implementation
- Develop a specialized scanner for Chinese securities with:
  - Market cap filtering
  - Liquidity requirements
  - Technical indicator calculations relevant to Chinese markets
  - Fundamental data integration if available
  - Risk metric calculations
- Ensure the scanner can handle different types of Chinese securities (A-shares, H-shares, ADRs)

### Step 3: Allocation Logic
- Implement portfolio allocation logic specific to Chinese securities:
  - Define maximum overall exposure to Chinese markets
  - Create rules for individual position sizing
  - Implement sector/industry diversification rules
  - Define correlation limits with other portfolio holdings
  - Create rebalancing triggers and logic

### Step 4: Integration with Trading System
- Connect the China scanner and allocation logic to the main trading system
- Ensure proper data flow between components
- Implement any necessary timing adjustments for market hours differences
- Add logging and monitoring specific to Chinese securities

### Step 5: Testing and Validation
- Test the scanner with historical data on Chinese securities
- Validate allocation decisions against predefined risk parameters
- Perform backtesting to evaluate performance
- Conduct paper trading tests before live implementation

### Step 6: Directory Setup
- **IMPORTANT**: As noted in the task, copy the implementation into a new directory
- Ensure all dependencies are properly maintained
- Document the new directory structure and purpose
- Update any import statements to reflect the new location

### Success Criteria
- Scanner successfully identifies promising Chinese securities based on predefined criteria
- Allocation logic properly sizes positions according to risk parameters
- Integration with the main trading system is seamless
- Implementation exists in its own directory as specified
- Performance metrics show improved returns with acceptable risk

## Resources
- Refer to existing scanner implementations in the codebase
- Research Chinese market structure and trading considerations
- Review Alpaca or other brokerage API documentation for trading Chinese securities
- Consider academic papers on factor investing in Chinese markets 