#!/usr/bin/env python
# coding: utf-8
"""
parallel_scan_v0_theta.py
=========================
Dwuwymiarowy skan (v0, theta) z dynamiczną liczbą atomów N(θ) oraz
równoległym wykonywaniem symulacji na wielu procesorach.
Dodane słupki błędów na wykresie a_min(theta).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit
from tqdm import tqdm
import csv
import multiprocessing as mp

import positronium_interferometer as psi
import monte_carlo_simulation as mc

# ============================================================================
# STAŁE PARAMETRY UKŁADU
# ============================================================================
L = 0.2
f_open = 0.5
C_max = 0.8
gap_to_stopper = 6e-3
n_points = 20
n_shots = 400
n_atoms_base = 680 * n_shots
theta_base = 5.5
V_SPREAD_ABS = 1.0e4
d_grating = psi.grating_period(mc.PS_23S_MARIAZZI.lam_L)

# ============================================================================
# ZAKRESY PRZESZUKIWANIA
# ============================================================================
v0_list = np.logspace(4.7, 5.4, 20)          # [m/s]
theta_mrad_list = np.array([0, 0.2, 0.5, 0.8, 1, 1.25, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.1, 2.25, 2.5, 3, 4, 5.5, 8])

# ============================================================================
# FUNKCJA POJEDYNCZEGO SKANU – zwraca teraz również błędy
# ============================================================================
def run_single_scan(v0, theta_mrad, seed_offset=2024):
    theta_rad = theta_mrad * 1e-3

    # Dynamiczna liczba atomów
    if theta_mrad == 0:
        n_atoms_current = int(n_atoms_base * (0.1 / theta_base))
    else:
        n_atoms_current = int(n_atoms_base * (theta_mrad / theta_base))
    n_atoms_current = max(n_atoms_current, 100)

    custom_beam = mc.BeamState(
        name=f"v0={v0:.1e}",
        lifetime=mc.PS_23S_MARIAZZI.lifetime,
        lam_L=mc.PS_23S_MARIAZZI.lam_L,
        v0=v0,
        v_spread=V_SPREAD_ABS
        # v_spread=v0/10
    )

    y_scan = np.linspace(0, 2 * d_grating, n_points)
    k_grating = 2 * np.pi / d_grating
    n_repeats = 10
    V_list = []
    N0_list = []

    for r in range(n_repeats):
        seed = seed_offset + int(v0 * 1e-4) * 1000 + int(theta_rad * 1e6) + r * 7
        sim_res = mc.simulate_grating3_scan_with_divergence(
            y_scan,
            beam=custom_beam,
            L=L,
            f_open=f_open,
            C_max=C_max,
            gap_to_stopper=gap_to_stopper,
            theta_div=theta_rad,
            n_atoms_per_point=n_atoms_current,
            seed=seed
        )
        y_data = sim_res["n_cross"]

        def fringe_model(y_pos, amp, vis, phase):
            return amp * (1.0 + vis * np.cos(k_grating * y_pos + phase))

        a_guess = np.mean(y_data)
        v_guess = (np.max(y_data) - np.min(y_data)) / (2.0 * a_guess if a_guess > 0 else 1.0)
        v_guess = np.clip(v_guess, 0.0, 1.0)

        try:
            popt, _ = curve_fit(
                fringe_model, y_scan, y_data,
                p0=[a_guess, v_guess, 0.0],
                bounds=([0, 0, -np.pi], [np.inf, 1.0, np.pi])
            )
            V_fit = popt[1]
        except Exception:
            V_fit = 0.0

        V_list.append(V_fit)
        N0_list.append(np.sum(y_data))

    V_mean = np.mean(V_list)
    V_std = np.std(V_list, ddof=1) if len(V_list) > 1 else 0.0
    N0_mean = np.mean(N0_list)
    N0_std = np.std(N0_list, ddof=1) if len(N0_list) > 1 else 0.0

    tau = L / v0
    if V_mean > 0 and N0_mean > 0:
        a_min = (1.0 / (V_mean * np.sqrt(N0_mean))) * (d_grating / (2 * np.pi)) * (1.0 / tau**2)
        # Propagacja błędu (przybliżenie liniowe, zakładamy niezależność V i N0)
        rel_err_V = V_std / V_mean if V_mean > 0 else 0.0
        rel_err_N0 = 0.5 * (N0_std / N0_mean) if N0_mean > 0 else 0.0
        a_min_err = a_min * np.sqrt(rel_err_V**2 + rel_err_N0**2)
    else:
        a_min = np.nan
        a_min_err = np.nan

    return (v0, theta_mrad, V_mean, V_std, N0_mean, N0_std, a_min, a_min_err)


# ============================================================================
# FUNKCJA OPAKOWUJĄCA (przyjmuje krotkę, pickle'owalna)
# ============================================================================
def worker(args):
    v0, theta_mrad, seed_offset = args
    return run_single_scan(v0, theta_mrad, seed_offset)


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=== Dwuwymiarowy skan (v0, theta) z multiprocessingiem ===")
    print(f"Liczba prędkości: {len(v0_list)}")
    print(f"Liczba rozbieżności: {len(theta_mrad_list)}")
    print(f"Bazowa liczba atomów (dla 5.5 mrad): {n_atoms_base}")
    print(f"Liczba punktów skanowania y: {n_points}")
    print(f"Powtórzenia wewnętrzne: 5")
    print("-" * 70)

    tasks = [(v0, theta) for v0 in v0_list for theta in theta_mrad_list]
    total = len(tasks)
    print(f"Łącznie kombinacji: {total}")

    n_cores = mp.cpu_count()
    print(f"Wykorzystanie {n_cores} rdzeni.")

    args_list = [(v0, theta, 2024) for v0, theta in tasks]

    try:
        results = []
        with mp.Pool(processes=n_cores) as pool:
            for res in tqdm(pool.imap_unordered(worker, args_list), total=total, desc="Symulacje"):
                results.append(res)

        results.sort(key=lambda x: (x[0], x[1]))

        # Zapis do CSV
        csv_file = "parallel_v0_theta_amin_with_errors.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["v0", "theta_mrad", "V_mean", "V_std", "N0_mean", "N0_std", "a_min", "a_min_err"])
            for row in results:
                writer.writerow(row)
        print(f"\nWyniki zapisano do: {csv_file}")

    except KeyboardInterrupt:
        csv_file = "parallel_v0_theta_amin_with_errors.csv"

        results = []

        with open(csv_file, "r", newline="") as f:
            reader = csv.DictReader(f)

            for row in reader:
                results.append([
                    float(row["v0"]),
                    float(row["theta_mrad"]),
                    float(row["V_mean"]),
                    float(row["V_std"]),
                    float(row["N0_mean"]),
                    float(row["N0_std"]),
                    float(row["a_min"]),
                    float(row["a_min_err"]),
                ])

        print(f"Wczytano {len(results)} wierszy z: {csv_file}")

    # ========================================================================
    # WYKRES 1: Rodzina krzywych a_min(theta) z słupkami błędów i paletą tab20
    # ========================================================================
    plt.figure(figsize=(12, 7))
    v0_unique = sorted(set(r[0] for r in results))
    
    # Pobierz mapę kolorów tab20
    cmap = plt.get_cmap('tab20')
    n_colors = len(v0_unique)

    for idx, v0 in enumerate(v0_unique):
        subset = [r for r in results if r[0] == v0 and not np.isnan(r[6])]
        if not subset:
            continue
        subset.sort(key=lambda x: x[1])
        theta_vals = [r[1] for r in subset]
        a_min_vals = [r[6] for r in subset]
        a_min_errs = [r[7] if not np.isnan(r[7]) else 0.0 for r in subset]
        label = f"v0 = {v0/1e5:.2f}×10⁵ m/s"
        # Wybierz kolor z mapy (indeks modulo 20)
        color = cmap(idx % 20)
        plt.errorbar(theta_vals, a_min_vals, yerr=a_min_errs, fmt='o-',
                     capsize=3, label=label, alpha=0.7, color=color)
    plt.xlabel("rozbieżność kątowa θ [mrad]", fontsize=12)
    plt.ylabel("$a_{min}$ [m/s²]", fontsize=12)
    plt.yscale('log')
    plt.title("Minimalne wykrywalne przyspieszenie vs θ dla różnych v0 (z błędami)")
    plt.grid(True, alpha=0.3, which='both')
    # Legenda – umieszczamy poza wykresem, aby nie zasłaniała danych
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8, ncol=1)
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # zostaw miejsce na legendę
    plt.savefig("parallel_amin_vs_theta_family_with_errors_tab20.png", dpi=150, bbox_inches='tight')
    plt.show()

    # ========================================================================
    # WYKRES 2: Mapa cieplna (bez zmian)
    # ========================================================================
    theta_list_sorted = sorted(set(r[1] for r in results))
    v0_list_sorted = sorted(v0_unique)
    Z = np.full((len(v0_list_sorted), len(theta_list_sorted)), np.nan)
    for i, v0 in enumerate(v0_list_sorted):
        for j, theta in enumerate(theta_list_sorted):
            val = next((r[6] for r in results if r[0] == v0 and r[1] == theta), np.nan)
            if not np.isnan(val) and val > 0:
                # Z[i, j] = np.log10(val)
                Z[i,j] = val

    z_min = np.nanmin(Z[Z > 0])
    z_max = np.nanmax(Z)

    fig, ax = plt.subplots(figsize=(10, 6))
    c = ax.pcolormesh(theta_list_sorted, np.array(v0_list_sorted)/1e5, Z,
                      shading='auto', cmap='plasma_r',
                      norm=LogNorm(vmin=z_min, vmax=z_max))
    ax.set_xlabel(r"beam divergence $\theta$ [mrad]", fontsize=12)
    ax.set_ylabel(r"mean velocity $v_0$ [×10⁵ m/s]", fontsize=12)
    ax.set_title(r"minimal detectable acceleration $a_{min}$ [m/s²]")
    cbar = plt.colorbar(c, ax=ax)
    cbar.set_label(r"$a_{min}$ [m/s²]", fontsize=16)
    cbar.ax.yaxis.set_tick_params(labelsize=14)
    CS = ax.contour(theta_list_sorted, np.array(v0_list_sorted)/1e5, Z,
                    levels=[-4, -3, -2, -1], colors='white', linewidths=0.5)
    ax.clabel(CS, inline=True, fontsize=8)
    plt.gca().xaxis.label.set_size(14)
    plt.gca().yaxis.label.set_size(14)
    plt.gca().title.set_size(14)
    plt.tick_params(labelsize=14)

    plt.tight_layout()
    plt.savefig("parallel_amin_heatmap.png", dpi=150)
    plt.show()

    print("Wykresy zapisano jako:")
    print(" - parallel_amin_vs_theta_family_with_errors_tab20.png")
    print(" - parallel_amin_heatmap.png")

if __name__ == "__main__":
    main()