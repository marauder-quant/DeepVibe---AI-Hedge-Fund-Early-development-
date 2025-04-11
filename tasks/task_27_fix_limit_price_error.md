# Task 27: Fix Sub-Penny Limit Price Error

## Description
Resolve the error occurring when placing buy orders with sub-penny limit prices that do not fulfill minimum pricing criteria.

## Background
The trading system is attempting to place orders with highly precise sub-penny limit prices that violate the broker's (Alpaca) pricing requirements. This is causing order rejection with error code 42210000.

Specific example from error log:
```
Error placing BUY order for 55 shares of ADPT: {"code":42210000,"message":"invalid limit_price 7.802249903678894. sub-penny increment does not fulfill minimum pricing criteria"}
[2025-04-11 00:11:27] Finished checking ADPT
```

## Objectives
1. Identify where and how limit prices are calculated
2. Implement proper price rounding to comply with minimum pricing criteria
3. Test fix across various price points to ensure compatibility
4. Add validation logic to prevent similar errors in the future

## Instructions for AI Agent

### Step 1: Code Analysis
- Review order placement code in the trading bot to identify where limit prices are calculated
- Determine the pricing algorithm that generated the invalid price (7.802249903678894)
- Analyze order placement functions and their interaction with the Alpaca API

### Step 2: Research Pricing Requirements
- Review Alpaca API documentation for limit price formatting requirements
- Determine the proper price rounding rules:
  - For stocks priced under $1.00: can be rounded to 4 decimal places (0.0001)
  - For stocks priced $1.00 and above: must be rounded to 2 decimal places (0.01)

### Step 3: Implement Price Formatting
- Add a price formatting function that properly rounds limit prices according to regulatory requirements
- Consider implementing the following logic:
  ```python
  def format_limit_price(price):
      if price < 1.0:
          return round(price, 4)  # For sub-$1 stocks, round to 4 decimal places
      else:
          return round(price, 2)  # For $1+ stocks, round to 2 decimal places
  ```
- Apply this formatting function to all limit order price calculations

### Step 4: Error Handling
- Add validation to catch and handle potential pricing errors before submitting orders
- Implement logging for price adjustments to track when rounding occurs
- Create appropriate error messages for prices that cannot be properly formatted

### Step 5: Testing
- Test the fix with multiple stock prices, especially around the $1.00 threshold
- Verify orders are accepted for stocks at various price points
- Create unit tests to ensure price formatting remains correct with future code changes

### Success Criteria
- No further occurrences of error code 42210000
- All limit orders are submitted with properly formatted prices
- Price rounding follows regulatory requirements
- Orders for both sub-$1 and $1+ stocks are processed correctly

## Resources
- Review Alpaca API documentation for latest pricing requirements
- Check the trading execution code that interacts with Alpaca
- Look for any price calculation logic in the strategy implementation 