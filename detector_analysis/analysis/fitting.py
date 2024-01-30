import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from scipy.special import erfc
from scipy.optimize import curve_fit

# ---------- import spectrum --------------------

def load_single_event(dir, key, events):
    """Loads a single event from a file and plots the spectrum."""
    file = f"{dir}/{key}.txt"
    if os.path.isfile(file):
        events[key] = np.loadtxt(file, skiprows=1, unpack=True)

def load_events(date, dir_list):
    """Loads the events from a list of directories and returns a dictionary with the events."""
    events = {}
    if type(dir_list) == str:
        dir_list = [dir_list]
    for dir in dir_list:
        for key in date.keys():
            load_single_event(dir, key, events)
    return events

# -------------- plot spectrum --------------------

def plot_raw_spectrum(ax, element, events, tot = False):
    """Plots the raw spectrum of the element."""
    ax.plot(*events[element][:2], linestyle = '-', label = element, linewidth = 1.5, alpha = 0.9)
    ax.grid(linestyle='dotted')
    ax.set_yscale('log')
    if tot:
        ax.set_xlabel('TOT / ticks')
    else:
        ax.set_xlabel('energy / keV')
    ax.set_ylabel('counts per hour')


def plot_average(ax, events, title, n, i):
    """Plots the average of the events with a moving average of n bins."""
    cmap = plt.get_cmap('tab20')
    i *= 2
    events[title] = events[title].T[events[title][0] > 5].T
    avg = np.convolve(events[title][1], np.ones(n)/n, mode='valid')
    ax.plot(events[title][0][n-1:] - n/2, avg, label = f'{title}: Average per ' + str(n) + ' keV', zorder = 10, color = cmap(i))
    ax.errorbar(*events[title][:2], fmt='.', yerr = events[title][2], label=f'{title}: Raw Data', ecolor = cmap(i+1), zorder = 1, color = cmap(i+1))
    ax.set_title(title)
    ax.set_xlabel('energy / keV')
    ax.set_ylabel('counts per hour')
    ax.legend()

def plot_error(ax, events, title):
    """Plots the counts with errorbars."""
    ax.errorbar(*events[title][:2], yerr = events[title][2], fmt = ".", ecolor = "r", capsize = 2, alpha = 0.8)
    ax.set_title(title)
    ax.set_xlabel("energy / keV")
    ax.set_ylabel("counts per hour")

def H_gauss(A, sig):
    return A/(np.sqrt(2*np.pi)*sig)

def gaussian(x, mu, sig, A):
    return H_gauss(A, sig)*np.exp(-np.power(x - mu, 2) / (2 * np.power(sig, 2)))

def erfunc(x, mu, sig, A, H):
    return H*H_gauss(A, sig)*erfc((x-mu)/(sig*np.sqrt(2)))

def tail(x, mu, sig, A, H_S, T):
    return H_S*H_gauss(A, sig)*np.exp((x-mu)/(T*sig))*erfc((x-mu)/(sig*np.sqrt(2)) + 1/(np.sqrt(2)*T))

def linear(x, d):
    return x*0 + d

def single(x, mu, sig, A, H, H_S, T, d):
    return gaussian(x, mu, sig, A) + erfunc(x, mu, sig, A, H) + tail(x, mu, sig, A, H_S, T) + linear(x, d)

def double(x, mu1, sig1, A1, H1, mu2, sig2, A2, H2, d):
    return gaussian(x, mu1, sig1, A1) + erfunc(x, mu1, sig1, A1, H1) + gaussian(x, mu2, sig2, A2) + erfunc(x, mu2, sig2, A2, H2) + linear(x, d)

def fit_gaussian(tot, counts, counts_err, mask):
    """Function to fit a single peak of a part of the spectrum. Returns the parameters and the covariance matrix, and the mask of the data used for the fit."""

    tot = tot[mask]
    counts = counts[mask]
    counts_err = counts_err[mask]

    p0 = [np.mean(tot), np.std(tot), np.sum(counts), 0,  0, 1, 0]
    bounds = ([np.min(tot), 1, 1, 0, 0, 0.1, 0], [np.max(tot), 200, np.inf, 1, 1, 1E3, np.percentile(counts, 50).mean()])

    popt, pcov = curve_fit(single, tot, counts, sigma = counts_err, absolute_sigma=True, p0=p0, bounds = bounds, maxfev=10000)

    return popt, pcov, mask

def fit_double_gaussian(tot, counts, counts_err, mask, mask1, mask2):
    """Function to fit a double peak of a part of the spectrum. Returns the parameters and the covariance matrix, and the mask of the data used for the fit."""

    p0 = [np.mean(tot[mask1]), np.std(tot[mask1]), np.sum(counts[mask1]), np.sum(tot[mask1])*0.001, np.mean(tot[mask2]), np.std(tot[mask2]), np.sum(counts[mask2]), np.sum(tot[mask2])*0.001, 0]
    bounds = ([np.min(tot[mask1]), 1, 0, 0, np.min(tot[mask2]), 1, 0, 0, 0], [np.max(tot[mask1]), 100, np.inf, np.inf, np.max(tot[mask2]), 100, np.inf, np.inf, np.percentile(counts[np.logical_or(mask1, mask2)], 50).mean() ])
    popt, pcov = curve_fit(double, tot[mask], counts[mask], sigma=counts_err[mask], absolute_sigma=True, p0= p0, bounds=bounds, maxfev=10000)
    return popt, pcov, mask

def get_a(e0, popt, error):
    if len(popt) > 3 and e0 > 50:
        a = popt[2]*(1 + popt[4])
        aerr = error[2] # np.sqrt((error[2]*(1 + popt[4]))**2 + (error[4]*popt[2])**2)
    else:
        a = popt[2]
        aerr = error[2]
    return a, aerr

def append_dict(cal_dict, e0, popt, pcov):
    """Function to append the calibration parameters to the calibration dictionary."""
    error = np.sqrt(np.diag(pcov))
    cal_dict['e0'].append(e0)
    cal_dict['energy'].append(popt[0])
    cal_dict['energy_err'].append(error[0])
    cal_dict['sigma'].append(popt[1])
    cal_dict['sigma_err'].append(error[1])
    a, a_err = get_a(e0, popt, error)
    cal_dict['A'].append(a)
    cal_dict['A_err'].append(a_err)

def do_double_gaussian_fit(peak_dict, tot, counts, counts_err, element, i):
    """Function to fit a double peak in the spectrium. Returns the parameters and the covariance matrix, and the mask of the data used for the fit."""
    mask = np.logical_and(tot >= min(peak_dict[element]['lower'][i]), tot <= max(peak_dict[element]['upper'][i]))
    mask1 = np.logical_and(tot >= peak_dict[element]['lower'][i][0], tot <= peak_dict[element]['upper'][i][0])
    mask2 = np.logical_and(tot >= peak_dict[element]['lower'][i][1], tot <= peak_dict[element]['upper'][i][1])
    return fit_double_gaussian(tot, counts, counts_err, mask, mask1, mask2)

def do_single_gaussian_fit(peak_dict, tot, counts, counts_err, element, i):
    """Function to fit a single peak in the spectrium. Returns the parameters and the covariance matrix, and the mask of the data used for the fit."""
    mask = np.logical_and(tot >= peak_dict[element]['lower'][i], tot <= peak_dict[element]['upper'][i])
    return fit_gaussian(tot, counts, counts_err, mask)

def fit_all(element, events, peak_dict):
    """Function to fit all peaks in the spectrum specified in the peak_dict. Returns the calibration dictionary."""

    calib_dict = {
     'e0':[],
     'energy':[],
     'energy_err':[],
     'sigma':[],
     'sigma_err':[],
     'A':[],
     'A_err':[]
     }
    
    tot, counts, counts_err = events[element]
    for i, energy in enumerate(peak_dict[element]['e0']):
        if isinstance(energy, tuple):
            popt, pcov, mask = do_double_gaussian_fit(peak_dict, tot, counts, counts_err, element, i)
            append_dict(calib_dict, energy[0], popt[:3], pcov[:3,:3])
            append_dict(calib_dict, energy[1], popt[4:7], pcov[4:7, 4:7])
        else:
            popt, pcov, mask = do_single_gaussian_fit(peak_dict, tot, counts, counts_err, element, i)
            append_dict(calib_dict, energy, popt, pcov)
    return pd.DataFrame(data = calib_dict)

def label_gaussian(e0, popt, pcov):
    """Creates a label for the fitted gaussian. Returns a string."""
    error = np.sqrt(np.diag(pcov))
    a, aerr = get_a(e0, popt, error)
    return r"$E_{peak}$ = " + f"{e0} keV\n" + r"$\mu$" + f" = {popt[0]:.2f}" + r"$\pm$" + f" {error[0]:.1f}\n" + r"$\sigma$"+f" = {popt[1]:.1f} " + r"$\pm$" + f" {error[1]:.1f}\n"+r"$A_{tot}$"+f" = {a:.0f} " + r"$\pm$" + f" {aerr:.0f}"

def label_double_gaussian(e01, e02, popt, pcov):
    """Creates a label for the fitted double gaussian. Returns a string."""
    error = np.sqrt(np.diag(pcov))
    return r"$E_{peak,1}$ = " + f"{e01} keV\n"+r"$\mu_1$"+f" = {popt[0]:.2f} " + r"$\pm$" + f" {error[0]:.1f}\n"+r"$\sigma_1$"+f" = {popt[1]:.1f} " + r"$\pm$" + f" {error[1]:.1f}\n$A_1$ = {popt[2]:.0f} " + r"$\pm$" + f" {error[2]:.0f}\n\n"+r"$E_{peak,2}$ = " + f"{e02} keV\n"+r"$\mu_2$"+f" = {popt[4]:.2f} " + r"$\pm$" + f" {error[4]:.1f}\n"+r"$\sigma_2$"+f" = {popt[5]:.1f} " + r"$\pm$" + f" {error[5]:.1f}\n$A_2$ = {popt[6]:.0f} " + r"$\pm$" + f" {error[6]:.0f}"
    
def plot_fit(element, events, peak_dict):
    """Function to plot the spectrum with the fitted peaks."""
    tot, counts, counts_err = events[element]
    fig, ax = plt.subplots()
    ax.plot(tot, counts, '+')
    for i, energy in enumerate(peak_dict[element]['e0']):
        if isinstance(energy, tuple):
            popt, pcov, mask = do_double_gaussian_fit(peak_dict, tot, counts, counts_err, element, i)
            ax.plot(tot[mask], double(tot[mask], *popt), label=label_double_gaussian(energy[0], energy[1], popt, pcov) + '\n')
        else:
            popt, pcov, mask = do_single_gaussian_fit(peak_dict, tot, counts, counts_err, element, i)
            ax.plot(tot[mask], single(tot[mask], *popt), label=label_gaussian(energy, popt, pcov) + '\n')
    ticks = np.arange(0, np.max(tot)+100, 100)
    ax.set_ylim(1, np.max(counts)*1.3)
    ax.set_xlim(ticks[0], ticks[-1])
    ax.set_yscale('log')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.2), ncol=3)
    ax.set_xticks(ticks)
    ax.grid(linestyle = 'dotted')
    ax.set_xlabel('energy / keV')
    ax.set_ylabel('counts per hour')
    ax.set_title(element)
    return fig, ax

def chi2_red(y, yfit, sigma, dof):
    """Calculates the reduced chi squared. Returns a float."""
    return np.sum((y - yfit)**2 / sigma**2)/dof

def plot_single_fit(element, events, peak_dict, energy_index):
    """Plots the fit of a single peak.
    
    Parameters
    ----------
    element : str
        Element to be fitted.
    events : dict
        Dictionary of events.
    peak_dict : dict
        Dictionary of peaks.
    energy_index : int
        Index of the energy to be fitted.
    """
    
    tot, counts, counts_err = events[element]
    energy = peak_dict[element]['e0'][energy_index]

    fig, ax = plt.subplots()

    if isinstance(energy, tuple):
        print('double')
        popt, pcov, mask = do_double_gaussian_fit(peak_dict, tot, counts, counts_err, element, energy_index)
        tfine = np.linspace(tot[mask][0], tot[mask][-1], 1000)
        ax.errorbar(tot[mask], counts[mask], fmt = '+', label='data', yerr =  counts_err[mask], color = 'dodgerblue',ecolor='red', capsize=1.5, elinewidth=0.8, capthick=1, zorder = 1, alpha = 0.8)
        ax.plot(tfine, double(tfine, *popt), label='fit', color = 'black', alpha = 1)
        ax.plot(tfine, gaussian(tfine, *popt[:3]) + gaussian(tfine, *popt[4:7]), label='gaussian', linestyle = 'dashdot', color = 'black')
        ax.plot(tfine, erfunc(tfine, *popt[:4]) + erfunc(tfine, *popt[4:8]), label='step', linestyle='dotted', color = 'black', alpha = 1)
        chi2 = chi2_red(counts[mask], double(tot[mask], *popt), np.sqrt(counts[mask]), len(tot[mask])-len(popt))
        ax.text(0.1, 0.9, label_double_gaussian(*energy, popt, pcov) + '\n' + r"$\chi^2_{red}$ = " + 
                 f"{round(chi2, 2)}", transform=plt.gca().transAxes, horizontalalignment='left', verticalalignment='top', bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 3})
    else:
        print('single')
        popt, pcov, mask = do_single_gaussian_fit(peak_dict, tot, counts, counts_err, element, energy_index)
        tfine = np.linspace(tot[mask][0], tot[mask][-1], 1000)
        ax.errorbar(tot[mask], counts[mask], fmt = '+', label='data', yerr =  counts_err[mask], color = 'dodgerblue',ecolor='red', capsize=1.5, elinewidth=0.8, capthick=1, zorder = 1, alpha = 0.8)
        ax.plot(tfine, single(tfine, *popt), label='fit', color = 'black', alpha = 1)
        ax.plot(tfine, gaussian(tfine, *popt[:3]), label='gaussian', linestyle = 'dashdot', color = 'black')
        ax.plot(tfine, tail(tfine, *popt[:3], *popt[4:-1]), label='tail', linestyle='dashed', color = 'black')
        ax.plot(tfine, erfunc(tfine, *popt[:4]), label='step', linestyle='dotted', color = 'black', alpha = 1)
        chi2 = chi2_red(counts[mask], single(tot[mask], *popt), np.sqrt(counts[mask]), len(tot[mask])-len(popt))
        ax.text(0.1, 0.9, label_gaussian(energy, popt, pcov) + '\n' + r"$\chi^2_{red}$ = " + f"{round(chi2, 2)}", transform = plt.gca().transAxes, horizontalalignment='left', verticalalignment='top', bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 8})
    ax.plot(tfine, linear(tfine, popt[-1]), label='constant background', linestyle='-', color = 'grey', zorder = 0)    
    print('fit parameters = ', popt)
    print('error', np.sqrt(np.diag(pcov)))
    ax.set_xlabel('energy / keV')
    ax.set_ylabel('counts per hour')
    ax.legend()
    #ax.set_title(f'{element}, {energy} keV')
    ax.set_title(f'{element}' + r", $E_{peak}$" + f" = {energy} keV") #, $\chi^2_{red} = $' + f"{round(chi2, 2)}")
    ax.set_ylim(0, np.max(counts[mask])*1.2)
    ax.set_xlim(tot[mask][0], tot[mask][-1])

    return fig, ax