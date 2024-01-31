import numpy as np
import matplotlib.pyplot as plt
import json
import sys
import pandas as pd
from datetime import datetime as dt
from configparser import ConfigParser

import gammaforge.analysis.fitting as fit
from gammaforge.utils.file_handling import save_plot

def get_date(config, isotope):
    """Get the date of calibration from the config file."""
    date = config.get(isotope, 'acquisition_date')
    date = dt.strptime(date, '%Y-%m-%d')
    return date

def get_distance(config, isotope):
    """Get the distance of the source from the detector."""
    distance = config.getfloat(isotope, 'distance')
    distance_error = config.getfloat(isotope, 'distance_error')
    return distance, distance_error

def load_date_dict(config, isotopes):
    """Load the date and distance dictionary from the config file."""
    date_dict = {}
    for isotope in isotopes:
        date_dict[isotope] = (get_date(config, isotope),
                                *get_distance(config, isotope))
    return date_dict

def split_peak_string(string):
    """Split the string of peaks into a list of floats and touples."""
    new_list = []
    for elm in string.strip("[] ").split(','):
        if '-' in elm:
            elm = elm.split('-')
            elm = (float(elm[0]), float(elm[1]))
        else:
            elm = float(elm)
        new_list.append(elm)
    return new_list

def get_peaks(config, isotope):
    """Get the peaks from the config file and load them into a dictionary."""
    peaks = {
        'e0': split_peak_string(config.get(isotope, 'peaks')),
        'fractions': json.loads(config.get(isotope, 'fractions')),
        'fractions_err': json.loads(config.get(isotope, 'fraction_errors')),
        'lower': split_peak_string(config.get(isotope, 'lower')),
        'upper': split_peak_string(config.get(isotope, 'upper')),
    }
    return peaks

def load_peak_dict(config, isotopes):
    """Load the peak dictionary from the config file."""
    peak_dict = {}
    for isotope in isotopes:
        peak_dict[isotope] = get_peaks(config, isotope)
    return peak_dict
    
def load_config():
    """Load the configuration file."""
    config = ConfigParser(inline_comment_prefixes='#')
    config.read('config.ini')
    isotopes = json.loads(config.get('isotopes', 'name'))
    date_dict = load_date_dict(config, isotopes)
    peak_dict = load_peak_dict(config, isotopes)
    dir = "..//" + config.get('isotopes', 'spectra_folder')
    return date_dict, peak_dict, dir

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
    date_dict, peak_dict, dir = load_config()
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