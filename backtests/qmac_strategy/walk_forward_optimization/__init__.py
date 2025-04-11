"""
Out-of-sample testing package for QMAC strategy.
"""

from backtests.qmac_strategy.walk_forward_optimization.oos_config import *
from backtests.qmac_strategy.walk_forward_optimization.oos_core import run_out_of_sample_test, run_out_of_sample_test_parallel
from backtests.qmac_strategy.walk_forward_optimization.oos_confidence import display_confidence_summary
from backtests.qmac_strategy.walk_forward_optimization.oos_database import get_best_oos_parameters_from_db 