import sys
sys.path.append(r"C:\Users\jonat\OneDrive - Universität Wien\4_COSMOS Seminar\gammaforge")
from gammaforge.utils.import_ini import load_config
from gammaforge.analysis.fitting import fit_all, load_events

def test_load_config():
    peaks, dates, dir = load_config(r"C:\Users\jonat\OneDrive - Universität Wien\4_COSMOS Seminar\gammaforge\example-scripts\config.ini")
    assert len(peaks["Ba133"]) == 3

def test_fit_all():
    dates, peaks, dir = load_config(r"C:\Users\jonat\OneDrive - Universität Wien\4_COSMOS Seminar\gammaforge\example-scripts\config.ini")
    events = load_events(dates, dir)
    df = fit_all("Ba133", events, peaks)
    assert len(df) == 4

    