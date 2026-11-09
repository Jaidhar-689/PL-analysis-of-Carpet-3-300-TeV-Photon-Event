# -*- coding: utf-8 -*-

# !pip install numpy pandas matplotlib scipy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy import optimize
import warnings
np.random.seed(21)

warnings.filterwarnings("ignore")

"""# Cosmology Constants & Functions

"""

H0_KM_S_MPC = 67.36
OMEGA_M = 0.3153
OMEGA_L = 0.6847
MPC_IN_KM = 3.0856775814913673e19
H0_INV_S = MPC_IN_KM / H0_KM_S_MPC


def _cosmo_integrand(zp):
    return (1.0 + zp) / np.sqrt(OMEGA_M * (1.0 + zp) ** 3 + OMEGA_L)


def compute_K1(z, E_high_o_GeV):
    integral, _ = quad(_cosmo_integrand, 0.0, z)
    K1_s_GeV = H0_INV_S * E_high_o_GeV / (1.0 + z) * integral
    return K1_s_GeV / 1e18

"""# Likelihood / chi2 definitions"""

def chi2_given_nuisance(nuisance_params, a_LV, data):
    alpha, mu, v = nuisance_params
    if v < 0:
        return 1e15

    dt, s_dt = data['dt'], data['s_dt']
    E, s_E = data['E'], data['s_E']
    K, s_K = data['K'], data['s_K']

    model_dt = a_LV * K + alpha * E + mu
    variance = (a_LV ** 2) * (s_K ** 2) + (alpha ** 2) * (s_E ** 2) + (v ** 2) + (s_dt ** 2)

    chi2 = np.sum((((dt - model_dt) ** 2) / variance) + np.log(2 * np.pi * variance))
    return chi2


def chi2_global(params, data):
    a_LV, alpha, mu, v = params
    if a_LV <= 0 or v < 0:
        return 1e15

    dt, s_dt = data['dt'], data['s_dt']
    E, s_E = data['E'], data['s_E']
    K, s_K = data['K'], data['s_K']

    model_dt = a_LV * K + alpha * E + mu
    variance = (a_LV ** 2) * (s_K ** 2) + (alpha ** 2) * (s_E ** 2) + (v ** 2) + (s_dt ** 2)

    return np.sum(((dt - model_dt) ** 2) / variance + np.log(2 * np.pi * variance))


def profile_chi2_fmin(data, a_LV, x0):
    res = optimize.fmin(
        chi2_given_nuisance,
        x0=x0,
        args=(a_LV, data),
        disp=False,
        full_output=True,
    )
    params_best, chi2_min = res[0], res[1]
    return chi2_min, params_best


def find_confidence_threshold(x_grid, y_grid, min_idx, target_chi2, direction='upper'):
    n = len(y_grid)
    if direction == 'upper':
        idx_range = range(min_idx, n - 1)
        step = 1
    elif direction == 'lower':
        idx_range = range(min_idx, 0, -1)
        step = -1
    else:
        raise ValueError("direction must be 'upper' or 'lower'")

    for i in idx_range:
        j = i + step
        lo, hi = min(y_grid[i], y_grid[j]), max(y_grid[i], y_grid[j])
        if lo <= target_chi2 <= hi:
            frac = (target_chi2 - y_grid[i]) / (y_grid[j] - y_grid[i])
            return x_grid[i] + frac * (x_grid[j] - x_grid[i])
    return None

"""# Profile-likelihood scan over E_LV (Nelder-Mead) + two-sided CIs

"""

def profile_scan_E(data, E_LV_grid, x0):
    n = len(E_LV_grid)
    chi2_vals = np.zeros(n)
    best_params = np.zeros((n, 3))
    nm_agree = np.zeros(n, dtype=bool)

    x0_alt = [x0[0] * 0.5 + 0.05, x0[1] * 0.5, x0[2] * 1.5 + 0.5]  # a different starting point

    for i, E_LV in enumerate(E_LV_grid):
        a_LV = 10.0 / E_LV  # self-inverse unit convention: a_LV[1e-18 GeV^-1] = 10 / E_LV[1e17 GeV]

        chi2_a, params_a = profile_chi2_fmin(data, a_LV, x0)
        chi2_b, params_b = profile_chi2_fmin(data, a_LV, x0_alt)

        if chi2_b < chi2_a:
            chi2_vals[i], best_params[i] = chi2_b, params_b
        else:
            chi2_vals[i], best_params[i] = chi2_a, params_a
        nm_agree[i] = abs(chi2_a - chi2_b) < 0.05

    min_idx = int(np.argmin(chi2_vals))
    delta_chi2 = chi2_vals - chi2_vals[min_idx]

    bounds_out = {}
    for label, target in [('1sigma', 1.0), ('95pct', 4.00)]:
        lo = find_confidence_threshold(E_LV_grid, delta_chi2, min_idx, target, direction='lower')
        hi = find_confidence_threshold(E_LV_grid, delta_chi2, min_idx, target, direction='upper')
        bounds_out[label] = (lo, hi)

    return delta_chi2, best_params, min_idx, bounds_out, nm_agree

"""# Data Processing & Dataset Assembly

"""

def build_datasets(data_path="data.csv"):
    df = pd.read_csv(data_path)
    df.columns = [c.strip() for c in df.columns]

    # Label I: 14 Multi-GeV Photons
    df_14 = df[df['Label'].str.strip() == 'I']
    dt_14, s_dt_14 = df_14['dt_obs'].to_numpy(), df_14['dt_obs_err'].to_numpy()
    E_14, s_E_14 = df_14['E_high_s'].to_numpy(), df_14['E_high_s_err'].to_numpy()
    K_14, s_K_14 = df_14['K1'].to_numpy(), df_14['K1_err'].to_numpy()

    # Label II: (99.3 GeV, 1.07 TeV, 12.2 TeV)
    df_3 = df[df['Label'].str.strip() == 'II']
    dt_3, s_dt_3 = df_3['dt_obs'].to_numpy(), df_3['dt_obs_err'].to_numpy()
    E_3, s_E_3 = df_3['E_high_s'].to_numpy(), df_3['E_high_s_err'].to_numpy()
    K_3, s_K_3 = df_3['K1'].to_numpy(), df_3['K1_err'].to_numpy()

    # Carpet-3 300 TeV Photon (GRB 221009A)
    z_carpet = 0.151
    E_high_o_carpet = 300.0e3
    E_high_o_err_carpet = 0.13 * E_high_o_carpet
    T_high_o_carpet = 4536.0
    T_low_o_carpet = 251.33
    dT_low_err_carpet = 5.00

    dt_carpet = np.array([(T_high_o_carpet - T_low_o_carpet) / (1.0 + z_carpet)])
    s_dt_carpet = np.array([dT_low_err_carpet / (1.0 + z_carpet)])
    E_carpet = np.array([E_high_o_carpet * (1.0 + z_carpet)])
    s_E_carpet = np.array([E_high_o_err_carpet * (1.0 + z_carpet)])
    K_carpet = np.array([compute_K1(z_carpet, E_high_o_carpet)])
    s_K_carpet = K_carpet * (E_high_o_err_carpet / E_high_o_carpet)

    datasets = {
        '14 Photons': {
            'dt': dt_14, 's_dt': s_dt_14, 'E': E_14, 's_E': s_E_14, 'K': K_14, 's_K': s_K_14
        },
        '14 Photons + Carpet-3': {
            'dt': np.concatenate((dt_14, dt_carpet)), 's_dt': np.concatenate((s_dt_14, s_dt_carpet)),
            'E': np.concatenate((E_14, E_carpet)), 's_E': np.concatenate((s_E_14, s_E_carpet)),
            'K': np.concatenate((K_14, K_carpet)), 's_K': np.concatenate((s_K_14, s_K_carpet))
        },
        '17 Photons': {
            'dt': np.concatenate((dt_14, dt_3)), 's_dt': np.concatenate((s_dt_14, s_dt_3)),
            'E': np.concatenate((E_14, E_3)), 's_E': np.concatenate((s_E_14, s_E_3)),
            'K': np.concatenate((K_14, K_3)), 's_K': np.concatenate((s_K_14, s_K_3))
        },
        '17 Photons + Carpet-3': {
            'dt': np.concatenate((dt_14, dt_3, dt_carpet)), 's_dt': np.concatenate((s_dt_14, s_dt_3, s_dt_carpet)),
            'E': np.concatenate((E_14, E_3, E_carpet)), 's_E': np.concatenate((s_E_14, s_E_3, s_E_carpet)),
            'K': np.concatenate((K_14, K_3, K_carpet)), 's_K': np.concatenate((s_K_14, s_K_3, s_K_carpet))
        }
    }
    return datasets

"""# Main execution: E_LV-grid profile-likelihood scan + plots

"""

datasets = build_datasets("data.csv")

# E_LV_grid = np.linspace(0.5, 15.0, 400)
E_LV_grid_GeV = np.logspace(5, 19, 500)

E_LV_grid = E_LV_grid_GeV / 1e17

x0_prof = [-0.15, -2.0, 5.0]  # initial guess for (alpha, mu, v)


scan_results = {}
for idx, (name, data) in enumerate(datasets.items()):
    delta_chi2, best_params, min_idx, ci, nm_agree = profile_scan_E(data, E_LV_grid, x0_prof)
    scan_results[name] = (E_LV_grid, delta_chi2, min_idx, ci, nm_agree)

    fig, axes = plt.subplots(figsize=(7, 5))
    # axes = axes.flatten()
    ax = axes
    ax.plot(E_LV_grid, delta_chi2, 'k-', lw=2, label='Profile Likelihood (Nelder-Mead)')
    ax.set_xscale('log')
    ax.set_xlim(1e0, 1e2)

    ax.grid(True, which='both', ls=':', alpha=0.3)
    ax.axhline(1.0, color='red', linestyle='--', alpha=0.6, label=r'$\Delta\chi^2=1$ (68.3%)')
    ax.axhline(4.00, color='blue', linestyle='--', alpha=0.6, label=r'$\Delta\chi^2=4.00$ (95%)')

    lo1, hi1 = ci['1sigma']
    lo95, hi95 = ci['95pct']
    for val in (lo1, hi1):
        if val is not None:
            ax.axvline(val, color='red', linestyle=':')
    for val in (lo95, hi95):
        if val is not None:
            ax.axvline(val, color='blue', linestyle=':')

    ax.set_ylim(-0.5, 12)
    ax.set_title(name, fontsize=13, fontweight='bold')
    ax.set_xlabel(r'$E_{LV} \ [10^{17} \ \mathrm{GeV}]$', fontsize=12)
    ax.set_ylabel(r'$\Delta\chi^2$', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f"profile_likelihood_carpet3_Egrid{idx}.png", dpi=150)
    # plt.show()

"""# Global best fit (Nelder-Mead, cross-checked with a second start point)

"""

x0_global = [3.3, -0.15, -2.0, 5.0]                 # [a_LV, alpha, mu, v]
x0_global_alt = [5.0, -0.05, 0.0, 3.0]               # alternate starting point

results_list = []
for name, data in datasets.items():
    res_a = optimize.fmin(chi2_global, x0=x0_global, args=(data,), disp=False, full_output=True)
    res_b = optimize.fmin(chi2_global, x0=x0_global_alt, args=(data,), disp=False, full_output=True)

    params_a, chi2_a = res_a[0], res_a[1]
    params_b, chi2_b = res_b[0], res_b[1]

    best_params_g, chi2_min_g = (params_a, chi2_a) if chi2_a <= chi2_b else (params_b, chi2_b)
    a_LV_best, alpha_best, mu_best, v_best = best_params_g

    chi2_agree = abs(chi2_a - chi2_b) < 0.05
    physical = (a_LV_best > 0) and (v_best > 0)
    agree = chi2_agree and physical

    E_LV_best = 10.0 / a_LV_best

    # Attach profile-likelihood 1-sigma bounds on E_LV computed above
    E_LV_grid_i, delta_chi2_i, min_idx_i, ci, nm_agree = scan_results[name]
    lo1, hi1 = ci['1sigma']  # numeric (float) or None

    e_lo_str = f"{lo1:.2f}" if lo1 is not None else "n/a (one-sided)"
    e_hi_str = f"{hi1:.2f}" if hi1 is not None else "n/a (one-sided)"

    sigma_minus = f"{E_LV_best - lo1:.2f}" if lo1 is not None else "n/a"
    sigma_plus = f"{hi1 - E_LV_best:.2f}" if hi1 is not None else "n/a"

    results_list.append({
        'Case Scenario': name,
        'a_LV [1e-18 GeV^-1]': f"{a_LV_best:.2f}",
        'E_LV [1e17 GeV]': f"{E_LV_best:.2f}",
        'E_LV 1sigma range': f"[{e_lo_str}, {e_hi_str}]",
        '-sigma / +sigma': f"[-{sigma_minus}, +{sigma_plus}]",
        'alpha [s/GeV]': f"{alpha_best:.2f}",
        'mu [s]': f"{mu_best:.2f}",
        'v [s]': f"{v_best:.2f}",
        'NM global': abs(chi2_a - chi2_b),
    })

df_results = pd.DataFrame(results_list)
print(df_results.to_markdown(index=False))


