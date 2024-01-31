import sys
sys.path.append(r"C:\Users\jonat\OneDrive - Universität Wien\4_COSMOS Seminar\gammaforge")
from gammaforge.analysis.detector_analysis import log_eff, resolution
import numpy as np


def test_log_eff():
    assert log_eff(1, 2, 3) == 2 + 3*np.log(1)

def test_resolution():
    assert resolution(5, 2) == 2*2*np.sqrt(2*np.log(2))/5