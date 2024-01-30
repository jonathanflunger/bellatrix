import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from matplotlib.ticker import StrMethodFormatter

from gammaforge.analysis.detector_analysis import combine_new_key, log_eff, eff_energy
from gammaforge.analysis.fitting import chi2_red, load_events
from gammaforge.peak_fitting import load_config
from gammaforge.utils.file_handling import save_plot

def get_dicts():
    dates, peaks, dir = load_config()
    events = load_events(dates, dir)
    return dates, peaks, events

def get_df(dates, peaks, events):

    print(dates)

    df = pd.read_csv('calibration.csv', index_col=[0,1], sep='\t')
    df[['A_norm', 'A_norm_err']] = combine_new_key(df, "get_A_norm", peak_dict = peaks)
    df[['A_rel', 'A_rel_err']] = combine_new_key(df, "get_A_rel", peak_dict = peaks)
    df[['abs_eff', 'abs_eff_err']] = combine_new_key(df, "get_eff", eff_func = "relative_eff", events_dict = events)
    df[['int_eff', 'int_eff_err']] = combine_new_key(df, "get_eff", eff_func = "intrinsic_eff", date=dates)
    df = df.dropna()
    print(df.head(10))

    return df

def fit_relative_efficiency(df):
    popt, pcov = curve_fit(log_eff, df.index.get_level_values(1), np.log(df['abs_eff']), sigma = df['abs_eff_err']/df['abs_eff'], absolute_sigma=True)

    return popt, pcov

def plot_eff_values(ax, df, eff_type):
    for index in df.index.get_level_values(0).unique():
        ax.errorbar(df.loc[index].index, df.loc[index][eff_type]*100, yerr = df.loc[index][eff_type + '_err']*100, fmt = 'o', label=f'{index}', markersize=4, zorder=10, ecolor = 'red', capsize=3, alpha = 1)

def plot_relative_efficiency(df, popt, pcov):

    efine = np.linspace(5, 1500, 1000)

    fig, ax = plt.subplots()

    ax.plot(efine, np.exp(log_eff(efine, *popt))*100, label=r'fit, $\ln\epsilon = a + b\ln E$', linestyle='dashed', color = 'black', alpha = 0.5)

    plot_eff_values(ax, df, 'abs_eff')

    ax.set_title('Relative Peak Efficiency')
    ax.set_xlabel('energy / keV')
    ax.set_ylabel(r'relative peak efficiency $\epsilon_\mathrm{rel,p}$')
    ax.set_xlim(70, 1500)
    ax.set_ylim(0.1, 200)
    ax.legend()

    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xticks([70, 100, 200, 300, 400, 700, 1000, 1500])
    ax.get_yaxis().set_major_formatter(StrMethodFormatter(r'{x:.1f} $\%$'))
    ax.get_xaxis().set_major_formatter(StrMethodFormatter(r'{x:.0f}'))
    ax.grid(which='both', axis = 'x')
    ax.grid(linestyle = 'dotted', which='minor', axis = "both", alpha = 0.5)
    ax.grid(linestyle = 'dashed', which='major', axis = "both", alpha = 0.7)

    return fig, ax, popt


def fit_instrinsic_efficiency(df):
    popt, pcov = curve_fit(log_eff, df.index.get_level_values(1), np.log(df['int_eff']), sigma = df['int_eff_err']/df['int_eff'], absolute_sigma=True)
    return popt, pcov

def plot_intrinsic_efficiency(df, popt, pcov, popt_rel):

    fig, ax = plt.subplots()

    a_eff = popt[0]
    a_eff_err = np.sqrt(np.diag(pcov))[0]
    b_eff = popt[1]
    b_eff_err = np.sqrt(np.diag(pcov))[1]

    efine = np.linspace(5, 1500, 1000)

    plot_eff_values(ax, df, 'int_eff')

    eff, eff_err = eff_energy(efine, popt, pcov)

    eff_err_neg = eff - eff_err
    eff_err_pos = eff + eff_err

    eff[eff > 100] = 100

    chi2 = chi2_red(df['int_eff'], np.exp(log_eff(df.index.get_level_values(1), *popt)), df['int_eff_err'], len(df['int_eff'].to_list())-len(popt))

    label = r'$\ln\epsilon = a + b\ln E$' + f"\na = {a_eff:.1f} \u00B1 {a_eff_err:.1f}\nb = {b_eff:.2f} \u00B1 {b_eff_err:.2f}" + f"\n$\chi^2_\\mathrm{{red}}$ = {chi2:.2f}"

    ax.plot(efine, np.exp(log_eff(efine, *popt_rel))*100, label=r'relative peak efficiency', linestyle='dashdot', color = 'lightblue')
    ax.fill_between(efine, eff_err_neg, eff_err_pos, alpha=0.5, color = 'dodgerblue', zorder = 0, label = r"$1\sigma$ error")
    ax.plot(efine, eff, label=label, linestyle='dashdot', color = 'black')

    ax.set_title('Intrinsic Peak Efficiency')
    ax.set_xlabel(r'energy $E$ / keV')
    ax.set_ylabel(r'efficiency $\epsilon$')

    order = np.arange(len(df.index.get_level_values(0).unique())+3)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc = 'lower left')

    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xticks([50, 70, 100, 200, 300, 500, 700, 1000, 1500])
    ax.get_yaxis().set_major_formatter(StrMethodFormatter(r'{x:.1f} $\%$'))
    ax.get_xaxis().set_major_formatter(StrMethodFormatter(r'{x:.0f}'))

    ax.set_ylim(0.1, 120)
    ax.set_xlim(70, 1500)
    ax.grid(which='both', axis = 'x')
    ax.grid(linestyle = 'dotted', which='minor', axis = "both", alpha = 0.5)
    ax.grid(linestyle = 'dashed', which='major', axis = "both", alpha = 0.7)

    return fig, ax
    
def main():
    df = get_df(*get_dicts())
    popt, pcov = fit_relative_efficiency(df)
    fig, ax, popt_rel = plot_relative_efficiency(df, popt, pcov)
    plt.show()
    save_plot(fig, "relative_efficiency")

    popt, pcov = fit_instrinsic_efficiency(df)
    fig, ax = plot_intrinsic_efficiency(df, popt, pcov, popt_rel)
    plt.show()
    save_plot(fig, "intrinsic_efficiency")
    
if __name__ == "__main__":
    main()