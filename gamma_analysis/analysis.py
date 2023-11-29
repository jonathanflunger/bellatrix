import tpx_analysis as tpx
import spectra_analysis as spec
from datetime import datetime as dt

DATE = {
    'Na22': (dt(2023, 7, 4), 163, 5),
    'Ba133': (dt(2023, 7, 24), 165, 5),
    'Cs137': (dt(2023, 8, 2), 178, 5),
    }

DIR = "../example-spectra"

def main():
    events = tpx.load_events(DATE, DIR)
    print(events)
    
if __name__ == "__main__":
    main()