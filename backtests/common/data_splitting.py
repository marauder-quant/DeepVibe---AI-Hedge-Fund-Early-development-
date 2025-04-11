"""
Data splitting configurations for backtesting.

This module provides consistent data splitting configurations that can be used
across different strategy implementations. It clearly separates parameters for
in-sample (IS) and out-of-sample (OOS) data.
"""

from enum import Enum
from typing import Dict, Any, Tuple, List, Optional, Union

# =====================================================================
# Splitting Method Enums
# =====================================================================

class SplitMethod(str, Enum):
    """Methods for splitting data into in-sample and out-of-sample periods."""
    ROLLING = "rolling"            # Moving window of fixed size
    EXPANDING = "expanding"        # Growing window with fixed start date
    ANCHORED = "anchored"          # Fixed training start with moving end
    WALK_FORWARD = "walk_forward"  # Combined expanding/rolling approach


# =====================================================================
# Default Configuration Sets
# =====================================================================

# Standard Walk-Forward Optimization (WFO) setup
DEFAULT_WFO_CONFIG = {
    "method": SplitMethod.WALK_FORWARD,
    "n_splits": 10,                  # Number of train/test splits
    "is_window_len": 365,            # Days in the in-sample window
    "oos_window_len": 100,           # Days in the out-of-sample window
    "expand_is_window": False,       # Keep fixed IS window size
    "overlap_windows": False,        # Non-overlapping windows
    "min_is_samples": 252,           # Minimum samples needed for IS window
    "min_oos_samples": 63,           # Minimum samples needed for OOS window
    "enforce_minimum_samples": True, # Skip split if doesn't meet minimums
}

# Standard Cross-Validation setup (multiple IS periods, one OOS period)
DEFAULT_CV_CONFIG = {
    "method": SplitMethod.EXPANDING,
    "n_splits": 5,                   # Number of folds
    "is_window_len": 252,            # Trading days (1 year) per in-sample
    "oos_window_len": 126,           # Trading days (6 months) for out-of-sample
    "expand_is_window": True,        # Use expanding window
    "overlap_windows": False,        # No overlap
    "min_is_samples": 126,           # Minimum 6 months for training
    "min_oos_samples": 21,           # Minimum 1 month for testing
    "enforce_minimum_samples": True, # Skip if doesn't meet minimums
}

# Fast testing setup (minimal splits for quick experiments)
FAST_TEST_CONFIG = {
    "method": SplitMethod.ROLLING,
    "n_splits": 3,                   # Few splits for quick testing
    "is_window_len": 252,            # 1 year in-sample
    "oos_window_len": 63,            # 3 months out-of-sample
    "expand_is_window": False,       # Fixed window size
    "overlap_windows": False,        # No overlap
    "min_is_samples": 126,           # Minimum 6 months
    "min_oos_samples": 21,           # Minimum 1 month
    "enforce_minimum_samples": True, # Skip if doesn't meet minimums
}

# Comprehensive setup (more granular splits for detailed analysis)
COMPREHENSIVE_CONFIG = {
    "method": SplitMethod.WALK_FORWARD,
    "n_splits": 20,                  # Many splits
    "is_window_len": 504,            # 2 years in-sample
    "oos_window_len": 63,            # 3 months out-of-sample
    "expand_is_window": True,        # Expanding window
    "overlap_windows": False,        # No overlap
    "min_is_samples": 252,           # Minimum 1 year
    "min_oos_samples": 21,           # Minimum 1 month
    "enforce_minimum_samples": True, # Skip if doesn't meet minimums
}

# =====================================================================
# Configuration Functions
# =====================================================================

def get_split_config(config_name: str = "default") -> Dict[str, Any]:
    """
    Get a predefined split configuration by name.
    
    Args:
        config_name: Name of the configuration to retrieve
        
    Returns:
        Dictionary containing split configuration parameters
    """
    config_map = {
        "default": DEFAULT_WFO_CONFIG,
        "wfo": DEFAULT_WFO_CONFIG,
        "cv": DEFAULT_CV_CONFIG,
        "fast": FAST_TEST_CONFIG,
        "comprehensive": COMPREHENSIVE_CONFIG
    }
    
    if config_name.lower() not in config_map:
        raise ValueError(f"Unknown configuration: {config_name}. " 
                         f"Available options: {', '.join(config_map.keys())}")
    
    return config_map[config_name.lower()].copy()

def create_custom_split_config(
    method: Union[SplitMethod, str] = SplitMethod.WALK_FORWARD,
    n_splits: int = 10,
    is_window_len: int = 365,
    oos_window_len: int = 100,
    expand_is_window: bool = False,
    overlap_windows: bool = False,
    min_is_samples: Optional[int] = None,
    min_oos_samples: Optional[int] = None,
    enforce_minimum_samples: bool = True
) -> Dict[str, Any]:
    """
    Create a custom data splitting configuration.
    
    Args:
        method: Method to use for splitting data
        n_splits: Number of splits to create
        is_window_len: Length of the in-sample window (in periods)
        oos_window_len: Length of the out-of-sample window (in periods)
        expand_is_window: Whether to use expanding in-sample windows
        overlap_windows: Whether to allow overlapping windows
        min_is_samples: Minimum samples required for in-sample window
        min_oos_samples: Minimum samples required for out-of-sample window
        enforce_minimum_samples: Whether to enforce minimum sample requirements
        
    Returns:
        Dictionary containing split configuration parameters
    """
    # Use reasonable defaults for minimum samples if not specified
    if min_is_samples is None:
        min_is_samples = max(int(is_window_len * 0.5), 20)
        
    if min_oos_samples is None:
        min_oos_samples = max(int(oos_window_len * 0.5), 5)
    
    # If method is a string, convert to SplitMethod
    if isinstance(method, str):
        try:
            method = SplitMethod(method.lower())
        except ValueError:
            raise ValueError(f"Invalid split method: {method}. " 
                             f"Available options: {', '.join([m.value for m in SplitMethod])}")
    
    return {
        "method": method,
        "n_splits": n_splits,
        "is_window_len": is_window_len,
        "oos_window_len": oos_window_len,
        "expand_is_window": expand_is_window,
        "overlap_windows": overlap_windows,
        "min_is_samples": min_is_samples,
        "min_oos_samples": min_oos_samples,
        "enforce_minimum_samples": enforce_minimum_samples
    }

def convert_to_vectorbt_params(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert the configuration to parameters compatible with vectorbt's splitter.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary compatible with vectorbt's split_kwargs
    """
    vbt_params = {}
    
    # Map method to left_to_right parameter
    if config["method"] in [SplitMethod.EXPANDING, SplitMethod.WALK_FORWARD]:
        vbt_params["left_to_right"] = False
    else:
        vbt_params["left_to_right"] = True
    
    # Set number of splits
    vbt_params["n"] = config["n_splits"]
    
    # Set window length
    vbt_params["window_len"] = config["is_window_len"]
    
    # Set out-of-sample length
    vbt_params["set_lens"] = (config["oos_window_len"],)
    
    # Additional parameters can be added as needed
    
    return vbt_params 