# Task 29: Create Master Requirements.txt with Categorized Dependencies

## Objective
Create a single, comprehensive requirements.txt file in the main directory that includes all necessary packages organized by category for a fresh cloud server installation.

## Requirements
- Single requirements.txt file in root directory
- All dependencies categorized and clearly labeled
- Specific version requirements for each package
- Clear section headers and comments
- Complete coverage of all dependencies

## Implementation Steps

1. **Audit Current Dependencies**
   - Review all Python files for imports
   - Identify all required packages
   - Document system-level dependencies
   - Check for version conflicts
   - Categorize packages into:
     * Core dependencies
     * Data processing
     * API integrations
     * Database
     * Testing
     * Development tools
     * Optional features

2. **Create Master requirements.txt**
   - Add clear section headers with comments
   - Organize packages by category
   - Specify exact version requirements
   - Add explanatory comments for complex dependencies
   - Include platform-specific notes where needed
   - Format example:
     ```
     # Core Dependencies
     numpy==1.24.3
     pandas==2.0.3
     
     # API Integrations
     alpaca-trade-api==3.0.0
     requests==2.31.0
     
     # Database
     sqlalchemy==2.0.23
     psycopg2-binary==2.9.9
     
     # Testing
     pytest==7.4.3
     pytest-cov==4.1.0
     
     # Development Tools
     black==23.11.0
     flake8==6.1.0
     
     # Optional Features
     # Uncomment for additional functionality
     # matplotlib==3.8.2
     # seaborn==0.13.0
     ```

3. **Update README.md**
   - Add section explaining the requirements.txt structure
   - Document installation command
   - Include troubleshooting for common installation issues
   - Add notes about optional dependencies

## Testing
- Test installation on fresh cloud server using the master requirements.txt
- Verify all core dependencies are correctly installed
- Test optional feature installation
- Check for version conflicts
- Verify all categories are properly installed

## Success Criteria
- Single requirements.txt file exists in root directory
- All dependencies are properly categorized
- Each package has specific version requirements
- Installation works on fresh cloud server
- No missing dependencies during runtime
- Clear documentation of optional features

## Notes
- Use exact version numbers (==) for all dependencies
- Include platform-specific notes where necessary
- Document any known compatibility issues
- Consider adding a simple installation script that handles the requirements.txt 