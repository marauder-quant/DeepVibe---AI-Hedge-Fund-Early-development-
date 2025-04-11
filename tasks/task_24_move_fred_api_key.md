# Task 24: Move FRED API Key to .env File

## Description
Securely relocate the FRED API key from its current location to the .env file for improved security and configuration management.

## Background
The Federal Reserve Economic Data (FRED) API key is currently hardcoded or stored in an insecure location within the codebase. This poses a security risk and makes configuration management more difficult. Moving the API key to the .env file follows best practices for credential management and configuration.

## Objectives
1. Identify all instances where the FRED API key is currently stored or used
2. Create or update the .env file to include the FRED API key
3. Modify code to reference the API key from the environment variables
4. Implement proper error handling for missing API keys
5. Update documentation to reflect the new configuration approach

## Instructions for AI Agent

### Step 1: Locate Current API Key Usage
- Search the codebase for "FRED", "fred_api", "api_key", etc. to find all instances of the FRED API key
- Identify all files that use the FRED API directly
- Note the current format and usage pattern of the API key
- Document all locations where changes will be needed
- Check if there are any existing environment variable utilities in the codebase

### Step 2: Examine and Update .env File
- Check if a .env file already exists in the project root
- If it exists, verify its format and current contents
- Add a new entry for the FRED API key with an appropriate variable name (e.g., `FRED_API_KEY=your_api_key_here`)
- If no .env file exists, create one following proper formatting
- Add the .env file to .gitignore if it's not already excluded
- Create a template .env.example file with placeholder values for documentation

### Step 3: Update Code to Use Environment Variable
- Modify all instances where the FRED API key is used to reference the environment variable instead
- Use appropriate environment variable loading:
  ```python
  import os
  from dotenv import load_dotenv
  
  load_dotenv()  # Load variables from .env file
  fred_api_key = os.getenv("FRED_API_KEY")
  ```
- Ensure the dotenv package is included in requirements.txt
- Implement consistent access pattern across all files

### Step 4: Add Error Handling and Validation
- Add validation to check if the FRED API key is present in environment variables
- Implement clear error messages when the API key is missing
- Add logging for API key status (found/not found) at application startup
- Create a function to validate the API key's format if possible
- Implement graceful degradation if the API key is invalid or missing

### Step 5: Testing and Documentation
- Test all functionality that uses the FRED API to ensure it works with the new configuration
- Update all relevant documentation to reflect the new approach
- Add setup instructions for configuring the FRED API key in the README
- Document the environment variable names and formats
- Create a troubleshooting section for API key issues

### Success Criteria
- FRED API key is completely removed from the codebase and stored only in .env
- All code successfully uses the API key from environment variables
- Appropriate error handling exists for missing or invalid API keys
- Documentation is updated with clear instructions for API key configuration
- No functionality is broken by the changes

## Resources
- Look for existing environment variable handling in the codebase
- Review the .env file for current structure and patterns
- Check for other API keys that might also need similar treatment
- Research Python best practices for environment variable usage 