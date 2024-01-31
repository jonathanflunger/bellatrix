import sys
sys.path.append(r"C:\Users\jonat\OneDrive - Universität Wien\4_COSMOS Seminar\gammaforge")
from gammaforge.utils.import_ini import load_config

def test_load_config():
    peaks, dates, dir = load_config(r"C:\Users\jonat\OneDrive - Universität Wien\4_COSMOS Seminar\gammaforge\example-scripts\config.ini")
    assert len(peaks["Ba133"]) == 3