"""
analyze_gravity_data.py (with phase unwrapping)
===============================================
Reads the CSV files, uses g=0 as reference.
Handles phase wrap-around by comparing to expected phase shift.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import positronium_interferometer as psi
import monte_carlo_simulation as mc


def fringe_model(y, N0, C, phi, d):
    return N0 * (1 + C * np.cos(2 * np.pi * y / d + phi))


def fit_fringe_from_data(y, n_cross, d, sigma=None, phi_guess=0.0):
    if sigma is None:
        sigma = np.sqrt(np.maximum(n_cross, 1))
    p0 = [np.mean(n_cross), 0.3, phi_guess]
    # Wider range to allow phases beyond ±π
    bounds = ([0, 0, -np.inf], [np.inf, 1, np.inf])
    popt, pcov = curve_fit(
        lambda y, N0, C, phi: fringe_model(y, N0, C, phi, d),
        y, n_cross, p0=p0, sigma=sigma,
        absolute_sigma=True, bounds=bounds, maxfev=20000
    )
    perr = np.sqrt(np.diag(pcov))
    return popt, perr


def main():
    print("=" * 78)
    print("Analyzing gravity_shift_demo data – recovering g_eff")
    print("Reference: g = 0 (gravity switched off)")
    print("With phase unwrapping based on expected phase shift")
    print("=" * 78)

    # Load the reference (g=0)
    ref_file = "gravity_data_g0.csv"
    try:
        df_ref = pd.read_csv(ref_file)
    except FileNotFoundError:
        print(f"ERROR: {ref_file} not found. Run mariazzi_gravity_shift_demo.py first.")
        return

    beam = mc.PS_23S_MARIAZZI
    d = psi.grating_period(beam.lam_L)
    L = 0.2

    # Symulujemy profil prędkości z uwzględnieniem rozpadu, aby znaleźć efektywne k
    rng_calib = np.random.default_rng(42)
    v_sampled = mc.sample_velocities(5_000_000, beam.v0, beam.v_spread / beam.v0, rng_calib)
    transit_time = 2 * L / v_sampled
    decay_time = rng_calib.exponential(beam.lifetime, size=5_000_000)
    
    # Wybieramy tylko te cząstki, które przeżyły drogę do 3. siatki
    survived = decay_time > transit_time
    v_survived = v_sampled[survived]
    
    # Obliczamy efektywne <tau^2> dla zarejestrowanych atomów
    tau_sq_eff = np.mean((L / v_survived) ** 2)
    k = 2 * np.pi / d * tau_sq_eff  # Poprawny współczynnik k uwzględniający rozpad promieniotwórczy

    y_ref = df_ref['y_nm'].values * 1e-9
    n_ref = df_ref['n_cross'].values
    sigma_ref = df_ref['sigma_n_cross'].values

    popt_ref, perr_ref = fit_fringe_from_data(y_ref, n_ref, d, sigma_ref)
    phi_ref = popt_ref[2]
    phi_ref_err = perr_ref[2]
    g_ref = 0.0

    print(f"\nReference: {ref_file}")
    print(f"  phi_ref = {phi_ref:.4f} ± {phi_ref_err:.4f} rad")
    print(f"  g_ref   = {g_ref:.2e} m/s²\n")

    files = ["gravity_data_g0.csv", "gravity_data_g1.csv", "gravity_data_g5.csv", "gravity_data_g10.csv", "gravity_data_g100.csv", "gravity_data_g500.csv"]

    for fname in files:
        if not pd.io.common.file_exists(fname):
            print(f"  File {fname} not found, skipping.")
            continue

        df = pd.read_csv(fname)
        y = df['y_nm'].values * 1e-9
        n = df['n_cross'].values
        sigma = df['sigma_n_cross'].values
        g_input = df['g_input'].iloc[0]
        gm = df['gm'].iloc[0]

        # Predict expected phase shift relative to g=0
        delta_phi_exp = k * (g_input - g_ref)

        # Fit with initial guess: phi_ref + delta_phi_exp
        phi_guess = phi_ref + delta_phi_exp
        popt, perr = fit_fringe_from_data(y, n, d, sigma, phi_guess=phi_guess)
        phi = popt[2]
        phi_err = perr[2]

        # Compute raw difference
        delta_phi_raw = phi - phi_ref
        delta_phi_raw_err = np.sqrt(phi_err**2 + phi_ref_err**2)

        # Unwrap: find integer n such that delta_phi_raw + 2π*n is closest to delta_phi_exp
        n_wrap = int(round((delta_phi_exp - delta_phi_raw) / (2 * np.pi)))
        delta_phi = delta_phi_raw + 2 * np.pi * n_wrap
        # The error remains the same (uncorrelated with the integer shift)
        delta_phi_err = delta_phi_raw_err

        # Recover g
        g_eff = g_ref + delta_phi / k
        g_eff_err = delta_phi_err / abs(k)

        print(f"\n--- File: {fname} ---")
        print(f"  Input g = {g_input:.2e} m/s²  (={gm} × g_earth)")
        print(f"  Fitted phi = {phi:.4f} ± {phi_err:.4f} rad")
        print(f"  Expected Δφ = {delta_phi_exp:.4f} rad")
        print(f"  Raw Δφ = {delta_phi_raw:.4f} ± {delta_phi_raw_err:.4f} rad")
        print(f"  Unwrapped Δφ = {delta_phi:.4f} ± {delta_phi_err:.4f} rad  (n_wrap = {n_wrap})")
        print(f"  Recovered g_eff = {g_eff:.2e} ± {g_eff_err:.2e} m/s²")
        if gm != 0:
            print(f"  Ratio g_eff / g_input = {g_eff/g_input:.3f}")
            print(f"  Agreement (sigma) = {(g_eff - g_input)/g_eff_err:.2f}")

        # Plot
        y_fine = np.linspace(y.min(), y.max(), 300)
        model = fringe_model(y_fine, popt[0], popt[1], popt[2], d)
        fig, ax = plt.subplots(figsize=(6,4))
        ax.errorbar(df['y_nm'], n, yerr=sigma, fmt='o', label='data', ms=2)
        ax.plot(y_fine*1e9, model, '-', label='fit')
        ax.set_xlabel('y [nm]')
        ax.set_ylabel('n_cross')
        ax.set_title(f'Fit for g = {gm}×g_earth\n'
                     f'recovered g_eff = {g_eff:.2e} ± {g_eff_err:.2e} m/s²')
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"fit_{gm}g_unwrapped.png", dpi=150)
        print(f"  Saved plot: fit_{gm}g_unwrapped.png")


if __name__ == "__main__":
    main()