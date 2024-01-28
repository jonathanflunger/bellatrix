import numpy as np
import matplotlib.pyplot as plt

from analysis import fitting as fit
from analysis import spectra_analysis as spec

from datetime import datetime as dt

DATE = {
    'Na22': (dt(2023, 7, 4), 163, 5),
    'Ba133': (dt(2023, 7, 24), 165, 5),
    'Cs137': (dt(2023, 8, 2), 178, 5),
    }

DIR = r"..\example-spectra"

def main():
    events = fit.load_events(DATE, DIR)
    return events
    #fig, ax = plt.subplots()
    #for key in DATE.keys():
    #    fit.plot_raw_spectrum(ax, key, events)
    #plt.show()
    
if __name__ == "__main__":
    main()