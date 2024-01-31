import sys
from datetime import datetime as dt
sys.path.append(r"C:\Users\jonat\OneDrive - Universität Wien\4_COSMOS Seminar\gammaforge")
from gammaforge.analysis.detector_analysis import factory_res

def test_fit_all():
    rel1, rel2 = factory_res()
    assert rel1.any() < 100
    