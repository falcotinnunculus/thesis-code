#!/usr/bin/env python
# coding: utf-8
"""
particles_scan_v0_theta.py
==========================
Dwuwymiarowy skan (v0, theta) dla modelu trzech siatek.
Wyznacza a_min z widoczności prążków dla g=0.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit
from tqdm import tqdm
import csv
import multiprocessing as mp
import sys

# ============================================================================
# STAŁE PARAMETRY UKŁADU (z particles_loop_parallel-copy.py)
# ============================================================================
d = 40.0          # okres siatki [μm]
f = 1/3
a_slit = f * d    # szerokość szczeliny [μm]
L = 20.0e4        # odległość między siatkami [μm]
Ls = 2.0e4        # odległość do stoppera [μm]
T12 = 1.142       # czas życia [μs]
g = 0.0           # przyspieszenie = 0

# ============================================================================
# ZAKRESY PRZESZUKIWANIA (takie same jak w optimize_velocity.py)
# ============================================================================
v0_list = np.logspace(4.7, 5.4, 20)          # [m/s] (liczbowo równe μm/μs)
theta_mrad_list = np.array([0, 0.2, 0.5, 0.8, 1, 1.25, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2, 2.1, 2.25, 2.5, 3, 4, 5.5, 8, 10, 12, 14, 17, 20])

# ============================================================================
# PARAMETRY SYMULACJI
# ============================================================================
n_points = 200                # liczba offsetów
n_shots = 400                 # liczba powtórzeń (używana do wyznaczenia bazy)
n_atoms_base = 680 * n_shots  # bazowa liczba atomów (dla theta_base)
theta_base = 5.5              # referencyjna rozbieżność [mrad]
n_repeats = 5                 # wewnętrzne powtórzenia dla estymacji błędów

# ============================================================================
# FUNKCJE WEKTOROWE (z particles_loop_parallel-copy.py)
# ============================================================================
def trans_vec(x, a):
    res = np.zeros_like(x, dtype=float)
    res[(x > -a/2) & (x < a/2)] = 1.0
    res[np.isclose(x, -a/2) | np.isclose(x, a/2)] = 0.5
    return res

def periodic_vec(x, off=0.0, d=d, a=a_slit):
    xm = (x + d/2 + off) % d - d/2
    return trans_vec(xm, a)

def track_vec(alpha, x, v, g=0.0, x0=0.0, y0=0.0):
    return np.tan(alpha) * (x - x0) + y0 - (g * x**2) / (2 * v**2)


def ann_point_vec(angles, offset, v, g=0.0):
    gr1 = L
    gr2 = 2*L
    gr3 = 3*L
    stop = gr3 + Ls
    y1 = track_vec(angles, gr1, v, g)
    y2 = track_vec(angles, gr2, v, g)
    y3 = track_vec(angles, gr3, v, g)

    per1 = periodic_vec(y1, 0.0)
    per2 = periodic_vec(y2, 0.0)
    per3 = periodic_vec(y3, offset)

    blocked1 = (per1 == 0)
    blocked2 = (per2 == 0)
    blocked3 = (per3 == 0)
    blocked = np.column_stack((blocked1, blocked2, blocked3))

    any_blocked = np.any(blocked, axis=1)
    first_idx = np.argmax(blocked, axis=1)
    idx = np.where(any_blocked, first_idx, 3)

    x_choices = np.array([gr1, gr2, gr3, stop])
    x_out = x_choices[idx]

    y_stack = np.column_stack((y1, y2, y3, track_vec(angles, stop, v, g)))
    y_out = y_stack[np.arange(len(angles)), idx]
    return x_out, y_out

def decay_point_vec(angles, times, v, g=0.0):
    x = v * np.cos(angles) * times
    y = track_vec(angles, x, v, g)
    return x, y

def ad_point_vec(angles, times, offset, v, g=0.0):
    x_ann, y_ann = ann_point_vec(angles, offset, v, g)
    x_dec, y_dec = decay_point_vec(angles, times, v, g)
    ann_first = (x_ann < x_dec)
    x_final = np.where(ann_first, x_ann, x_dec)
    y_final = np.where(ann_first, y_ann, y_dec)
    return x_final, y_final

def simulate_offsets(angles, dec_times, offsets, v, g=0.0):
    """
    Dla każdego offsetu oblicza liczbę cząstek, które dotarły do stoppera.
    angles, dec_times: tablice o długości N (liczba cząstek)
    offsets: tablica offsetów
    Zwraca tablicę counts o długości len(offsets)
    """
    stop = 3*L + Ls
    counts = np.zeros(len(offsets), dtype=int)
    for i, off in enumerate(offsets):
        x_final, _ = ad_point_vec(angles, dec_times, off, v, g)
        counts[i] = np.sum(np.isclose(x_final, stop, atol=1e-9))
    return counts

# ============================================================================
# FUNKCJA POJEDYNCZEGO SKANU – dla danej pary (v0, theta_mrad)
# ============================================================================
def run_single_scan(v0, theta_mrad, seed_offset=2024):
    theta_rad = theta_mrad * 1e-3   # [rad]
    v = v0  # liczbowo μm/μs = m/s

    # Dynamiczna liczba atomów (jak w optimize_velocity)
    if theta_mrad == 0:
        n_atoms_current = int(n_atoms_base * (0.1 / theta_base))
    else:
        n_atoms_current = int(n_atoms_base * (theta_mrad / theta_base))
    n_atoms_current = max(n_atoms_current, 100)

    offsets = np.linspace(0, d, n_points)  # [μm]

    V_list = []
    N0_list = []
    k = 2 * np.pi / d

    def fringe_model(off, amp, vis, phase):
        return amp * (1.0 + vis * np.cos(k * off + phase))

    for rep in range(n_repeats):
        seed = seed_offset + int(v0 * 1e-4) * 1000 + int(theta_rad * 1e6) + rep * 1000
        np.random.seed(seed)
        # Generujemy cząstki raz dla wszystkich offsetów
        angles = np.random.uniform(-theta_rad/2, theta_rad/2, n_atoms_current)
        dec_times = np.random.exponential(T12 / np.log(2), n_atoms_current)
        # Obliczamy liczbę trafień dla każdego offsetu
        y_data = simulate_offsets(angles, dec_times, offsets, v, g=0.0)

        # Dopasowanie
        amp_guess = np.mean(y_data)
        vis_guess = (np.max(y_data) - np.min(y_data)) / (2.0 * amp_guess if amp_guess > 0 else 1.0)
        vis_guess = np.clip(vis_guess, 0.0, 1.0)

        try:
            popt, _ = curve_fit(fringe_model, offsets, y_data,
                                p0=[amp_guess, vis_guess, 0.0],
                                bounds=([0, 0, -np.pi], [np.inf, 1.0, np.pi]))
            V_rep = popt[1]
        except Exception:
            V_rep = 0.0
        V_list.append(V_rep)
        N0_list.append(np.sum(y_data))

    V_mean = np.mean(V_list)
    V_std = np.std(V_list, ddof=1) if len(V_list) > 1 else 0.0
    N0_mean = np.mean(N0_list)
    N0_std = np.std(N0_list, ddof=1) if len(N0_list) > 1 else 0.0

    # Obliczenie a_min
    tau = L / v0   # czas w μs (L w μm, v0 w μm/μs)
    if V_mean > 0 and N0_mean > 0:
        a_min_um_us2 = (1.0 / (V_mean * np.sqrt(N0_mean))) * (d / (2 * np.pi)) * (1.0 / tau**2)
        a_min = a_min_um_us2 * 1e6   # przeliczenie na m/s²
        rel_err_V = V_std / V_mean if V_mean > 0 else 0.0
        rel_err_N0 = 0.5 * (N0_std / N0_mean) if N0_mean > 0 else 0.0
        a_min_err = a_min * np.sqrt(rel_err_V**2 + rel_err_N0**2)
    else:
        a_min = np.nan
        a_min_err = np.nan

    return (v0, theta_mrad, V_mean, V_std, N0_mean, N0_std, a_min, a_min_err)

# ============================================================================
# FUNKCJA OPAKOWUJĄCA (dla multiprocessing)
# ============================================================================
def worker(args):
    v0, theta_mrad, seed_offset = args
    return run_single_scan(v0, theta_mrad, seed_offset)

# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=== Dwuwymiarowy skan (v0, theta) dla modelu trzech siatek ===")
    print(f"Liczba prędkości: {len(v0_list)}")
    print(f"Liczba rozbieżności: {len(theta_mrad_list)}")
    print(f"Bazowa liczba atomów (dla 5.5 mrad): {n_atoms_base}")
    print(f"Liczba punktów offset: {n_points}")
    print(f"Powtórzenia wewnętrzne: {n_repeats}")
    print("-" * 70)

    tasks = [(v0, theta) for v0 in v0_list for theta in theta_mrad_list]
    total = len(tasks)
    print(f"Łącznie kombinacji: {total}")

    n_cores = mp.cpu_count()
    print(f"Wykorzystanie {n_cores} rdzeni.")

    args_list = [(v0, theta, 2024) for v0, theta in tasks]

    # Ustawienie metody startowej tylko dla systemów innych niż Windows
    if sys.platform != 'win32':
        try:
            mp.set_start_method('fork', force=True)
        except RuntimeError:
            pass

    try:
        results = []
        with mp.Pool(processes=n_cores) as pool:
            for res in tqdm(pool.imap_unordered(worker, args_list), total=total, desc="Symulacje"):
                results.append(res)

        results.sort(key=lambda x: (x[0], x[1]))

        csv_file = "particles_scan_v0_theta_amin.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["v0", "theta_mrad", "V_mean", "V_std", "N0_mean", "N0_std", "a_min", "a_min_err"])
            for row in results:
                writer.writerow(row)
        print(f"\nWyniki zapisano do: {csv_file}")

    except KeyboardInterrupt:
        csv_file = "particles_scan_v0_theta_amin.csv"
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
    # WYKRES 1: Rodzina krzywych a_min(theta) z słupkami błędów
    # ========================================================================
    plt.figure(figsize=(12, 7))
    v0_unique = sorted(set(r[0] for r in results))
    cmap = plt.get_cmap('tab20')
    for idx, v0 in enumerate(v0_unique):
        subset = [r for r in results if r[0] == v0 and not np.isnan(r[6])]
        if not subset:
            continue
        subset.sort(key=lambda x: x[1])
        theta_vals = [r[1] for r in subset]
        a_min_vals = [r[6] for r in subset]
        a_min_errs = [r[7] if not np.isnan(r[7]) else 0.0 for r in subset]
        label = f"v0 = {v0/1e5:.2f}×10⁵ m/s"
        color = cmap(idx % 20)
        plt.errorbar(theta_vals, a_min_vals, yerr=a_min_errs, fmt='o-',
                     capsize=3, label=label, alpha=0.7, color=color)
    plt.xlabel("rozbieżność kątowa θ [mrad]", fontsize=12)
    plt.ylabel("$a_{min}$ [m/s²]", fontsize=12)
    plt.yscale('log')
    plt.title("Minimalne wykrywalne przyspieszenie vs θ dla różnych v0 (model 3 siatek)")
    plt.grid(True, alpha=0.3, which='both')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8, ncol=1)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig("particles_amin_vs_theta_family.png", dpi=150, bbox_inches='tight')
    plt.show()

    # ========================================================================
    # WYKRES 2: Mapa cieplna (heatmap)
    # ========================================================================
    theta_list_sorted = sorted(set(r[1] for r in results))
    v0_list_sorted = sorted(v0_unique)
    Z = np.full((len(v0_list_sorted), len(theta_list_sorted)), np.nan)
    for i, v0 in enumerate(v0_list_sorted):
        for j, theta in enumerate(theta_list_sorted):
            val = next((r[6] for r in results if r[0] == v0 and r[1] == theta), np.nan)
            if not np.isnan(val) and val > 0:
                Z[i, j] = val

    z_min = np.nanmin(Z[Z > 0])
    z_max = min(np.nanmax(Z),5e4)

    fig, ax = plt.subplots(figsize=(10, 6))
    c = ax.pcolormesh(theta_list_sorted, np.array(v0_list_sorted)/1e5, Z,
                      shading='auto', cmap='plasma_r',
                      norm=LogNorm(vmin=z_min, vmax=z_max))
    ax.set_xlim(0.5, None)
    ax.set_xlabel(r"beam divergence $\theta$ [mrad]", fontsize=12)
    ax.set_ylabel(r"mean velocity $v_0$ [×10⁵ m/s]", fontsize=12)
    ax.set_title(r"minimal detectable acceleration $a_{min}$ [m/s²]")
    cbar = plt.colorbar(c, ax=ax)
    cbar.set_label(r"$a_{min}$ [m/s²]")
    # Kontury (opcjonalnie)
    # levels = np.percentile(Z[~np.isnan(Z)], [10, 30, 50, 70, 90])
    # CS = ax.contour(theta_list_sorted, np.array(v0_list_sorted)/1e5, Z,
    #                 levels=levels, colors='white', linewidths=0.5)
    # ax.clabel(CS, inline=True, fontsize=8)
    plt.gca().xaxis.label.set_size(14)
    plt.gca().yaxis.label.set_size(14)
    plt.gca().title.set_size(14)
    plt.tick_params(labelsize=14)
    plt.tight_layout()
    plt.savefig("particles_amin_heatmap.png", dpi=150)
    plt.show()

    print("Wykresy zapisano jako:")
    print(" - particles_amin_vs_theta_family.png")
    print(" - particles_amin_heatmap.png")

if __name__ == "__main__":
    main()