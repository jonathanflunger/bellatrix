import json
from configparser import ConfigParser
from datetime import datetime as dt

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
    
def load_config(ini_filepath):
    """Load the configuration file."""
    config = ConfigParser(inline_comment_prefixes='#')
    config.read(ini_filepath)
    isotopes = json.loads(config.get('isotopes', 'name'))
    date_dict = load_date_dict(config, isotopes)
    peak_dict = load_peak_dict(config, isotopes)
    dir = "..//" + config.get('isotopes', 'spectra_folder')
    return date_dict, peak_dict, dir