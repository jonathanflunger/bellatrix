import sys
sys.path.append(r"C:\Users\jonat\OneDrive - Universität Wien\4_COSMOS Seminar\gammaforge")
from gammaforge.analysis.detector_analysis import log_eff
import numpy as np


def test_log_eff():
    assert log_eff(1, 2, 3) == 2 + 3*np.log(1)