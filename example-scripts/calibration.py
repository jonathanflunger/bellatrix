import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.ticker import MultipleLocator

from gammaforge.analysis.detector_analysis import linear, calibration
from gammaforge.analysis.fitting import chi2_red, load_events
from peak_fitting import load_config
from gammaforge.utils.file_handling import save_plot

def fit_calibration(df):
    """Fit the calibration function to the data."""
    popt, pcov = curve_fit(linear, df.index.get_level_values(1), df['energy'], sigma=df['energy_err'], absolute_sigma=True)
    a = round(popt[0],4)
    b = round(popt[1],2)
    aerr = round(np.sqrt(np.diag(pcov))[0],4)
    berr = round(np.sqrt(np.diag(pcov))[1],2)
    return a, b, aerr, berr

def plot_calibration(df, a, b, aerr, berr):
    """Plot the calibration function and the data."""

    fig, ax = plt.subplots()

    ax.errorbar(df.index.get_level_values(1), df['energy'], yerr = df['energy_err'], fmt = 'x', label='callibration data', markersize=5, color = 'black', zorder=10, ecolor='red', capsize=2)

    efine = np.linspace(5, 1600, 1000)

    cal_err = np.sqrt((aerr*efine)**2 + berr**2)
    chi2 = chi2_red(df['energy'], linear(df.index.get_level_values(1), a, b),
                     df['energy_err'], len(df['energy'])-2)

    ax.plot(efine, linear(efine, a, b), 
            label = r'linear calibration, $E_{uncal} = c_1 E_ + c_0$' +'\n' 
            + r'$c_1$' + f' = {a} \u00B1 {aerr}' +'\n' + r'$c_0$' 
            + f' = {b} \u00B1 {berr}' +'\n' + r'$\chi^2_{red}$' 
            + f' = {round(chi2,2)}', linestyle='dashed')
    ax.fill_between(efine, linear(efine, a, b) - 3*cal_err, linear(efine, a, b) + 3*cal_err, alpha=1, color='darkorange', label = r'$3\sigma$ error')

    ax.grid(linestyle = 'dotted')
    ax.set_ylabel('uncalibrated energy / a.u.')
    ax.set_xlabel('energy / keV')
    ax.set_title('Linear Calibration')
    ax.legend(loc='upper left')
    ax.set_xlim(0, 1600)
    ax.set_ylim(0, 1700)

    return fig, ax

def plot_calibrated_spectrum(df, a, b):

    fig, ax = plt.subplots()

    ticks = [0, 200, 800, 1000, 1400]
    date_dict, peak_dict, dir = load_config()
    events = load_events(date_dict, dir)

    for tick in ticks:
        ax.vlines(tick, 0, 1e5, linestyle='dotted', color = 'k', alpha = 0.2, linewidth = 0.5)

    for key in df.index.get_level_values(0).unique():
        energy, counts, counts_err = events[key]#[:,3:]
        energy = calibration(energy, a, b)
        ax.plot(energy, counts, label=key, linewidth=1.5, linestyle = "-", marker = 'none')
        for e0 in df.xs(key).index:
            if e0 > 350 or (e0 < 100 and e0 > 78):
                lim = 1e3 if e0 > 1000 else 1e5
                ax.vlines(e0, 0, lim, linestyle='--', linewidth = 0.8, color = 'black', alpha = 0.8)
                ticks.append(e0)

    ax.vlines(-1, 0, 0, linestyle='--', linewidth = 0.8, color = 'black', label = 'major peaks')
    ax.xaxis.set_minor_locator(MultipleLocator(50))
    ax.set_yscale('log')
    ax.set_xticks(sorted(ticks))
    ax.set_xlim(0, 1400)
    ax.set_ylim(1, 1e5)
    ax.set_xlabel('energy / keV')
    ax.set_ylabel('counts per hour')
    ax.set_title('Calibrated Spectrum')
    ax.legend(loc='upper right')

    ax.grid(linestyle = 'dotted', which='minor', axis = "x", alpha = 0.5)
    ax.grid(linestyle = 'dotted', which='major', axis = "y", alpha = 0.5)

    return fig, ax
    
def main():
    df = pd.read_csv('calibration.csv', index_col=[0,1], sep='\t')
    a, b, aerr, berr = fit_calibration(df)
    
    fig, ax = plot_calibration(df, a, b, aerr, berr)
    plt.show()
    save_plot(fig, 'calibration')

    fig, ax = plot_calibrated_spectrum(df, a, b)
    plt.show()
    save_plot(fig, 'calibrated_spectrum')

if __name__ == "__main__":
    main()