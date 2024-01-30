import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from bellatrix.analysis.detector_analysis import combine_new_key, res_func, get_res
from bellatrix.analysis.fitting import chi2_red
from bellatrix.utils.file_handling import save_plot

def get_df():
    df = pd.read_csv('calibration.csv', index_col=[0,1], sep='\t')
    df[['res', 'res_err']] = combine_new_key(df, "get_resolution")
    df = df.dropna()
    return df

def fit_resolution(df):
    popt, pcov = curve_fit(res_func, df.index.get_level_values(1), df['res']**2, sigma = df['res']*df['res_err']*2, absolute_sigma=True, bounds = ([0, 0, 0], [np.inf, np.inf, np.inf]))
    return popt, pcov

efine = np.linspace(5, 2000, 1000)

def label_isotopes(df):
    label = ""
    for isotope in df.index.get_level_values(0).unique():
        label += isotope + ", "
    return label[:-2]

def plot_resolution(df, popt, pcov):
    res_fit, res_err = get_res(efine, popt, pcov)

    _a, _b, _c, = popt
    _aerr, _berr, _cerr = np.sqrt(np.diag(pcov))

    print(f"paramter a: {_a:.3f} \u00B1 {_aerr:.3f}")
    print(f"paramter b: {_b:.3f} \u00B1 {_berr:.3f}")
    print(f"paramter c: {_c:.3f} \u00B1 {_cerr:.3f}")

    chi2 = chi2_red(df['res']**2, res_func(df.index.get_level_values(1), *popt), df['res']*df['res_err']*2, len(df['res'])-len(popt))

    fig, ax = plt.subplots()

    ax.plot(efine, res_fit, label=r'$R = \sqrt{a^2E^{-2} + b^2E^{-1} + c^2}$' 
            + f'\n$\chi^2_{{red}} = {chi2:.2f}$',
            alpha=1, linestyle='dashdot', zorder=20, color = 'black')
    ax.fill_between(efine, res_fit - 3*res_err, res_fit + 3*res_err, alpha=0.5, color='dodgerblue', zorder=0, label=r"$3\sigma$ error")

    ax.plot(efine, _a/efine*100, color = 'grey', label = r"$aE^{-1}$, " + f'a = ({_a:.2f} \u00B1 {_aerr:.2f}) keV', linestyle='-', alpha=1)
    ax.plot(efine, _b/np.sqrt(efine)*100, color = 'grey', label = r"$bE^{-1/2}$, " + f'b = ({_b:.2f} \u00B1 {_berr:.2f})' + r" $\sqrt{\mathrm{keV}}$", linestyle='--', alpha=1)
    ax.hlines(_c*100, 0, 1600 , color = 'grey', label = f'c = {_c:.4f} \u00B1 {_cerr:.4f}', linestyle='dotted', alpha=1)

    ax.errorbar(df.index.get_level_values(1), df['res']*100, yerr=df['res_err']*100, fmt='.', label = label_isotopes(df), markersize=10, color='blue', zorder=10, ecolor='red', capsize=3)

    ax.set_xlabel(r'energy $E$ / keV')
    ax.set_ylabel(r'resolution $R$ / %')
    ax.set_title(r'Relative Resolution $R = FWHM/E$')
    ax.set_xlim(0, 1600)
    ax.set_yticks(np.arange(0, 31, 5))
    ax.set_ylim(0, 25)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[-1:] + handles[:-1], labels[-1:] + labels[:-1], loc='upper right')
    ax.minorticks_on()
    ax.grid(linestyle='dotted', which='minor', axis="both", alpha=0.5)
    ax.grid(linestyle='dashed', which='major', axis="both", alpha=0.7)

    return fig, ax

def main():
    df = get_df()
    popt, pcov = fit_resolution(df)
    fig, ax = plot_resolution(df, popt, pcov)
    plt.show()
    save_plot(fig, "resolution")

if __name__ == "__main__":
    main()
