import sys
sys.path.append(r"\Users\jonat\OneDrive - Universität Wien\4_COSMOS Seminar\bellatrix")
sys.path.append(r"\Users\jonat\OneDrive - Universität Wien\4_COSMOS Seminar\bellatrix\detector_analysis")
sys.path.append(r"\Users\jonat\OneDrive - Universität Wien\4_COSMOS Seminar\bellatrix\detector_analysis\analysis")

from detector_analysis.analysis import fitting as fit

from datetime import datetime as dt

DIR = r"C:\Users\jonat\OneDrive - Universität Wien\4_COSMOS Seminar\bellatrix\example-spectra"

DATE = {
    'Na22': (dt(2023, 7, 4), 163, 5),
    'Ba133': (dt(2023, 7, 24), 165, 5),
    'Cs137': (dt(2023, 8, 2), 178, 5),
    }

def test_load_events():
    events = fit.load_events(DATE, DIR)
    assert len(events) == 3