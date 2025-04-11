# Task 22: Fix Kill Switch Warning Message

## Description
Correct the kill switch functionality to properly handle remaining positions and provide accurate warning messages.

## Background
The trading system's kill switch currently displays inaccurate warning messages about remaining positions when activated. This issue needs to be fixed to ensure proper shutdown procedures and accurate position reporting when the kill switch is triggered.

## Objectives
1. Identify the root cause of incorrect "WARNING: remaining positions" messages
2. Fix the position detection and reporting logic
3. Improve the kill switch shutdown sequence
4. Enhance logging and notification during kill switch activation
5. Add proper validation of kill switch completion

## Instructions for AI Agent

### Step 1: Issue Investigation
- Locate the kill switch implementation in the codebase
- Review the position tracking and reporting mechanism
- Identify why the system reports remaining positions incorrectly
- Analyze the shutdown sequence and its interaction with position tracking
- Check for any race conditions or timing issues in the process

### Step 2: Position Detection Fix
- Update the position detection logic to accurately identify open positions
- Implement proper account balance and position reconciliation
- Ensure all position types (long, short, options, etc.) are correctly detected
- Add validation checks to confirm position status
- Fix any data synchronization issues between local and broker data

### Step 3: Warning Message Correction
- Modify the warning message generation to accurately report remaining positions
- Implement different message levels based on position status
- Add detailed position information to warning messages
- Ensure messages are clear and actionable
- Create conditional messaging based on kill switch trigger reasons

### Step 4: Shutdown Sequence Enhancement
- Review and improve the kill switch shutdown sequence
- Implement proper order of operations for position checking and reporting
- Add confirmation steps for critical shutdown actions
- Create a more robust position handling mechanism during shutdown
- Ensure all resources are properly released during kill switch activation

### Step 5: Testing and Validation
- Create comprehensive test cases for the kill switch functionality
- Test with various position scenarios (no positions, partial positions, full positions)
- Validate warning messages are accurate in all test cases
- Test kill switch activation under different market conditions
- Verify logs contain proper information for post-mortem analysis

### Success Criteria
- Kill switch correctly identifies and reports remaining positions
- Warning messages are accurate and helpful
- Shutdown sequence executes in the correct order
- All positions are properly handled during kill switch activation
- Logs provide clear information about the kill switch process

## Resources
- Review trading bot shutdown procedures in the codebase
- Examine position tracking implementation
- Check logging and notification systems
- Investigate broker API interaction for position verification
- Look at existing error handling for kill switch activation 