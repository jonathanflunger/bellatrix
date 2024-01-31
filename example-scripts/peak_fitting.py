import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd

import gammaforge.analysis.fitting as fit
from gammaforge.utils.file_handling import save_plot
from gammaforge.utils.import_ini import load_config

def plot_raw_spectra(events):
    """Plot the raw spectra."""
    fig, ax = plt.subplots()

    for key in events.keys():
        fit.plot_raw_spectrum(ax, key, events)

    ax.set_xticks(np.arange(0, 1401, 100))
    ax.set_xlim(0, 1400)
    ax.set_ylim(1, 1E5)
    ax.legend(title = 'Isotopes', loc = 'upper right')
    ax.set_title('Raw Spectra, Pre-Calibrated Energy')

def plot_fit_peak(spectrum, events, peak_dict):
    for index in range(len(peak_dict[spectrum]['e0'])):
        fig, ax = fit.plot_single_fit(spectrum, events, peak_dict, index)
        ax.set_xlabel(r'a.u')
        plt.show()
        return fig, ax

def plot_fit_all_peaks(key, events, peak_dict):
        fig, ax = fit.plot_fit(key, events, peak_dict)
        plt.show()
        return fig, ax

def replace_err(df, key):
    df.loc[df[key+'_err'] >= df[key], key] = np.nan
    df.loc[df[key+'_err'] >= df[key], key+'_err'] = np.nan

def write_to_df(events, peak_dict):
    dfs = []
    for key in events.keys():
        dfs.append(fit.fit_all(key, events, peak_dict))
        print(f'{key} done')
    df = pd.concat(dfs, axis=0, keys=list(events.keys()), names=['Isotope'])
    df = df.set_index(['e0'], inplace=False, append=True).droplevel(1)
    for key in ['energy', 'sigma', 'A']:
        replace_err(df, key)
    return df

def save_df(df, filename):
    df.to_csv(filename, sep='\t', index=True, header=True)

def main():
    date_dict, peak_dict, dir = load_config("config.ini")
    events = fit.load_events(date_dict, dir)
    if len(sys.argv) > 1:
        if sys.argv[1] == 'plot':
            for key in events.keys():
                fig, ax = plot_fit_peak(key, events, peak_dict)
                save_plot(fig, key + '_peak_fit')
                fig, ax = plot_fit_all_peaks(key, events, peak_dict)
                save_plot(fig, key + '_all_peaks_fit')
    df = write_to_df(events, peak_dict)
    save_df(df, 'calibration.csv')
        
if __name__ == "__main__":
    main()