"""
monte_carlo_simulation.py
==========================
Particle-by-particle Monte Carlo simulation of the positronium Mach-Zehnder
interferometer from Oberthaler (2002), Nucl. Instr. Meth. Phys. Res. B 192,
129-134, built on top of `positronium_interferometer.py`.

Idea
----
Instead of using the closed-form fringe pattern Eq. (1) directly, this
module throws individual simulated positronium atoms at the apparatus.
Each atom independently:

    1. is born with a velocity v drawn from the beam's velocity
       distribution (mean v0, relative spread `v_spread_rel`);
    2. has a random lifetime drawn from Exp(lifetime) (ortho-Ps decays
       radioactively); it is lost if it decays before reaching the
       detector, i.e. before the full transit time 2L/v;
    3. if it survives, is registered by the detector with probability
       `detector_eff` (collimation / detection-efficiency losses);
    4. if detected, ends up in the "bright" or "dark" output port with
       quantum probability P = (1 +/- C_max*cos(phi))/2, where
       phi = 2*pi*dx/d + phi_g(v) is the *total* phase from Eqs. (1)-(2)
       -- note phi_g depends on the atom's own velocity, since
       tau = L / v.

Summing many atoms reproduces the analytic fringe pattern, but with
genuine Poissonian shot noise and -- if the velocity spread is large
enough -- a *reduction* of the apparent contrast coming purely from the
velocity-dependence of the gravitational phase (an effect the analytic
formulas do not capture, since Eq. (2) is written for a single tau).

Fitting the noisy simulated scan recovers the gravitational phase / `g`
with an uncertainty that can be checked, from first principles, against
the analytic sensitivity formula Eq. (3).
"""

from __future__ import annotations
import numpy as np
from scipy.optimize import curve_fit
from tqdm import tqdm

import positronium_interferometer as psi


# ---------------------------------------------------------------------------
# Atom-by-atom propagation
# ---------------------------------------------------------------------------
def sample_velocities(n, v0, v_spread_rel, rng):
    """Draw n atom velocities from a Gaussian beam velocity distribution
    with mean v0 and relative spread v_spread_rel (0 = monochromatic)."""
    if v_spread_rel <= 0:
        return np.full(n, v0)
    sigma_v = v_spread_rel * v0
    v = rng.normal(v0, sigma_v, size=n)
    return np.clip(v, 0.05 * v0, None)   # guard against an unphysical tail


def simulate_scan_point(dx, *, d, v0, v_spread_rel, lifetime, L, C_max,
                         g, detector_eff, phase_offset, n_atoms, rng):
    """Monte Carlo simulate n_atoms positronium atoms for a single
    third-grating displacement dx.

    Returns (n_bright, n_dark, n_lost, tau_mean_detected).
    """
    v = sample_velocities(n_atoms, v0, v_spread_rel, rng)
    tau = L / v                      # interaction time entering Eq. (2)
    transit_time = 2 * L / v         # full interferometer transit time

    decay_time = rng.exponential(lifetime, size=n_atoms)
    survives = decay_time > transit_time
    detected = survives & (rng.random(n_atoms) < detector_eff)

    phi_g = 2 * np.pi / d * g * tau ** 2                 # Eq. (2), per atom
    phi = 2 * np.pi * dx / d + phi_g + phase_offset       # Eq. (1) phase

    p_bright = np.clip((1 + C_max * np.cos(phi)) / 2.0, 0.0, 1.0)
    bright = detected & (rng.random(n_atoms) < p_bright)
    dark = detected & ~bright

    n_bright = int(np.sum(bright))
    n_dark = int(np.sum(dark))
    n_lost = int(n_atoms - np.sum(detected))
    tau_mean = float(np.mean(tau[detected])) if np.any(detected) else np.nan
    return n_bright, n_dark, n_lost, tau_mean


def simulate_scan(dx_array, *, state_key="2s", L=2.5, E_kin_eV=10.0,
                   v_spread_rel=0.0, C_max=0.8, g=psi.g_earth,
                   detector_eff=1.0, phase_offset=0.0,
                   n_atoms_per_point=2000, seed=None):
    """Monte Carlo simulate a full third-grating displacement scan
    dx_array, atom by atom.

    Parameters
    ----------
    state_key : "1s" or "2s"        -- which o-Ps transition/grating
    L : float [m]                   -- grating spacing (total length 2L)
    E_kin_eV : float                -- mean kinetic energy of the Ps beam
    v_spread_rel : float            -- relative (1 sigma) velocity spread
    C_max : float                   -- intrinsic (instrumental) contrast
    g : float [m/s^2]               -- gravitational acceleration to probe
    detector_eff : float            -- detection efficiency (0..1)
    phase_offset : float [rad]      -- extra calibration phase (default 0)
    n_atoms_per_point : int         -- simulated atoms per scan point
    seed : int or None              -- RNG seed for reproducibility

    Returns a dict with arrays n_bright, n_dark, n_lost, tau_mean indexed
    like dx_array, plus the grating period d, nominal velocity v0, etc.
    """
    rng = np.random.default_rng(seed)
    state = psi.PS_STATES[state_key]
    d = psi.grating_period(state.lam_L)
    v0 = psi.ps_velocity(E_kin_eV)

    dx_array = np.asarray(dx_array, dtype=float)
    n_bright = np.zeros(len(dx_array), dtype=int)
    n_dark = np.zeros(len(dx_array), dtype=int)
    n_lost = np.zeros(len(dx_array), dtype=int)
    tau_mean = np.zeros(len(dx_array))

    for i, dx in enumerate(dx_array):
        nb, nd, nl, tm = simulate_scan_point(
            dx, d=d, v0=v0, v_spread_rel=v_spread_rel,
            lifetime=state.lifetime, L=L, C_max=C_max, g=g,
            detector_eff=detector_eff, phase_offset=phase_offset,
            n_atoms=n_atoms_per_point, rng=rng)
        n_bright[i], n_dark[i], n_lost[i], tau_mean[i] = nb, nd, nl, tm

    return dict(dx=dx_array, n_bright=n_bright, n_dark=n_dark, n_lost=n_lost,
                tau_mean=tau_mean, d=d, v0=v0, L=L, lifetime=state.lifetime,
                state=state_key, C_max=C_max, g=g,
                n_atoms_per_point=n_atoms_per_point)


# ---------------------------------------------------------------------------
# Fitting the simulated fringe scan  (recovering Eq. 1's N0, C, phi)
# ---------------------------------------------------------------------------
def _fringe_model(dx, N0, C, phi, d):
    return N0 * (1 + C * np.cos(2 * np.pi * dx / d + phi))


def fit_fringe(scan):
    """Fit the simulated n_bright(dx) counts to the Eq. (1) fringe model.
    Returns (popt, perr) = ([N0, C, phi], [sigma_N0, sigma_C, sigma_phi]).
    """
    dx, n_bright, d = scan["dx"], scan["n_bright"], scan["d"]
    sigma = np.sqrt(np.maximum(n_bright, 1))  # Poisson (shot-noise) errors

    def model(dx_, N0, C, phi):
        return _fringe_model(dx_, N0, C, phi, d)

    p0 = [max(n_bright.mean(), 1.0), 0.5, 0.0]
    bounds = ([0, 0, -np.pi], [np.inf, 1, np.pi])
    popt, pcov = curve_fit(model, dx, n_bright, p0=p0, sigma=sigma,
                            absolute_sigma=True, bounds=bounds, maxfev=20000)
    perr = np.sqrt(np.diag(pcov))
    return popt, perr


def recovered_g(phi, phi_err, d, tau):
    """Invert Eq. (2) to turn a fitted phase (and its uncertainty) back
    into a gravitational acceleration estimate."""
    g_fit = phi * d / (2 * np.pi * tau ** 2)
    g_err = abs(phi_err) * d / (2 * np.pi * tau ** 2)
    return g_fit, g_err


# ===========================================================================
# Experimental realisation from Mariazzi, Caravita, Doser, Nebbia & Brusa,
# "Toward inertial sensing with a 2^3S positronium beam",
# Eur. Phys. J. D 74, 79 (2020).
#
# The key difference with respect to the idealised Oberthaler (2002) light-
# grating recombiner used above: the third grating is a *physical*
# (mechanical) absorbing grating of the same period d, mounted on a
# translation stage and scanned along y (Figs. 5-6 of the paper). An atom
# arriving at the grating-3 plane either:
#   - lands on an *open* slit and continues on to a "stopper" a short
#     distance further downstream, where it eventually annihilates
#     ("crossing" atoms, detected by the downstream gamma detector), or
#   - lands on a *solid* bar of the grating and annihilates right there
#     ("stopped" atoms, detected by a gamma detector at the grating plane).
# Both channels are simple gamma-ray counting detectors with no need for
# spatial resolution (Sec. 4 of the paper) -- scanning y and recording
# N_cross(y) and N_stop(y) maps out the interference pattern.
# ===========================================================================
from dataclasses import dataclass as _dataclass


@_dataclass
class BeamState:
    """A quasi-monochromatic Ps beam characterised directly by its mean
    velocity (as done in Mariazzi et al. 2020), rather than via a kinetic
    energy as in the Oberthaler-style PsState above."""
    name: str
    lifetime: float   # [s]
    lam_L: float        # standing-light-wave (1st & 2nd grating) wavelength [m]
    v0: float            # mean beam velocity [m/s]
    v_spread: float      # 1-sigma *absolute* velocity spread [m/s]


# Sec. 2 of the paper: "a velocity distribution with a mean of 1e5 m/s and
# a standard deviation of less than 1e4 m/s"; Sec. 3.2: lam = 1312 nm
# standing wave (same choice as Oberthaler 2002); lifetime 1142 ns.
PS_23S_MARIAZZI = BeamState(
    name="2^3S o-Ps (Mariazzi et al. 2020)",
    lifetime=1142e-9,
    lam_L=1312e-9,
    v0=1.0e5,
    v_spread=1.0e4,
)


def sample_fringe_phase_per_atom(phi_array, V, rng):
    """Vectorised rejection sampling of one fringe phase theta_i in
    [0, 2*pi) per atom, from the periodic interference probability density
        rho_i(theta) = (1/2pi) * (1 + V_i cos(theta + phi_i)),
    i.e. each atom may carry its own phase offset phi_i and visibility V_i.
    """
    phi_array = np.asarray(phi_array, dtype=float)
    V_is_array = isinstance(V, np.ndarray)
    
    n = len(phi_array)
    theta = np.empty(n)
    idx_remaining = np.arange(n)
    while idx_remaining.size > 0:
        m = idx_remaining.size
        cand = rng.uniform(0, 2 * np.pi, size=m)
        
        # Jeśli V jest tablicą (różny kontrast dla każdego atomu), wybierz odpowiednie elementy
        V_curr = V[idx_remaining] if V_is_array else V
        
        accept_prob = (1 + V_curr * np.cos(cand + phi_array[idx_remaining])) / (1 + V_curr)
        accept = rng.random(m) < accept_prob
        theta[idx_remaining[accept]] = cand[accept]
        idx_remaining = idx_remaining[~accept]
    return theta


def simulate_grating3_scan_point(y, *, d, f_open, beam: BeamState, L, C_max,
                                  g, phase_offset, gap_to_stopper, n_atoms, rng):
    """Monte Carlo simulate n_atoms positronium atoms arriving at the
    entrance of the interferometer, propagate them to the (physical) third
    grating positioned at y, and sort them into "crossing" (open slit,
    continuing to the stopper) or "stopped" (solid bar, annihilating on
    the grating) -- following Fig. 6 of Mariazzi et al. (2020).

    Returns (n_cross, n_stop, n_lost_before_grating3, n_lost_in_gap).
    """
    v = sample_velocities(n_atoms, beam.v0, beam.v_spread / beam.v0, rng)
    tau = L / v                          # Eq. (2) interaction time
    transit_to_g3 = 2 * L / v            # grating1 -> grating3

    decay_time = rng.exponential(beam.lifetime, size=n_atoms)
    reaches_g3 = decay_time > transit_to_g3
    n_lost_before = int(n_atoms - np.sum(reaches_g3))
    n_reach = int(np.sum(reaches_g3))
    if n_reach == 0:
        return 0, 0, n_lost_before, 0

    v_r = v[reaches_g3]
    tau_r = tau[reaches_g3]
    phi_g = 2 * np.pi / d * g * tau_r ** 2          # Eq. (2), per atom
    phi_total = phi_g + phase_offset

    theta = sample_fringe_phase_per_atom(phi_total, C_max, rng)

    # physical mask: open slit of angular width 2*pi*f_open, located at the
    # phase corresponding to the grating's mechanical position y
    y_phase = (2 * np.pi * y / d) % (2 * np.pi)
    rel = (theta - y_phase) % (2 * np.pi)
    crosses = rel < (2 * np.pi * f_open)
    n_stop = int(np.sum(~crosses))                  # annihilate on the grating

    n_cand = int(np.sum(crosses))
    if gap_to_stopper > 0 and n_cand > 0:
        v_cross = v_r[crosses]
        t_gap = gap_to_stopper / v_cross
        decay_gap = rng.exponential(beam.lifetime, size=n_cand)
        survive_gap = decay_gap > t_gap
        n_cross = int(np.sum(survive_gap))
        n_lost_gap = n_cand - n_cross
    else:
        n_cross = n_cand
        n_lost_gap = 0

    return n_cross, n_stop, n_lost_before, n_lost_gap


def simulate_grating3_scan(y_array, *, beam=PS_23S_MARIAZZI, L=0.2,
                            f_open=0.5, C_max=0.8, g=psi.g_earth,
                            phase_offset=0.0, gap_to_stopper=20e-3,
                            n_atoms_per_point=68000, seed=None):
    """Scan the physical third grating over y_array and record the
    "crossing" / "stopped" gamma counts at each position -- the Monte
    Carlo equivalent of Fig. 6 of Mariazzi et al. (2020).

    Parameters
    ----------
    beam : BeamState        -- e.g. PS_23S_MARIAZZI
    L : float [m]            -- grating spacing (paper: ~0.2 m, "L ~ v*2us")
    f_open : float            -- open (slit) fraction of the grating period
                                  (not stated explicitly in the paper; 0.5
                                  assumed by default)
    C_max : float             -- intrinsic (light-grating) fringe visibility
    g : float [m/s^2]         -- acceleration to probe
    gap_to_stopper : float [m]-- short distance between 3rd grating and the
                                  stopper (paper: 6 mm, <5% extra loss at
                                  v=1e5 m/s)
    n_atoms_per_point : int   -- simulated atoms entering the interferometer,
                                  per scan point (paper: ~680 per shot with
                                  5.5 mrad divergence; default here
                                  corresponds to ~100 accumulated shots)
    """
    rng = np.random.default_rng(seed)
    d = psi.grating_period(beam.lam_L)
    y_array = np.asarray(y_array, dtype=float)

    n_cross = np.zeros(len(y_array), dtype=int)
    n_stop = np.zeros(len(y_array), dtype=int)
    n_lost = np.zeros(len(y_array), dtype=int)
    n_gap = np.zeros(len(y_array), dtype=int)

    for i, y in tqdm(enumerate(y_array)):
        nc, ns, nlb, nlg = simulate_grating3_scan_point(
            y, d=d, f_open=f_open, beam=beam, L=L, C_max=C_max, g=g,
            phase_offset=phase_offset, gap_to_stopper=gap_to_stopper,
            n_atoms=n_atoms_per_point, rng=rng)
        n_cross[i], n_stop[i], n_lost[i], n_gap[i] = nc, ns, nlb, nlg

    return dict(y=y_array, n_cross=n_cross, n_stop=n_stop, n_lost=n_lost, n_gap=n_gap,
                d=d, L=L, beam=beam, f_open=f_open, C_max=C_max, g=g,
                n_atoms_per_point=n_atoms_per_point)


def fit_grating3_fringe(scan, channel="n_cross"):
    """Fit either the n_cross(y) or n_stop(y) curve from
    simulate_grating3_scan() to the Eq. (1) fringe model. Returns
    (popt, perr) = ([N0, C, phi], [errors]); C here is the *observed*
    visibility, which is reduced compared to C_max by the finite open
    fraction of the physical grating (a real, additional contrast loss
    not present for the idealised light-grating recombiner)."""
    y, counts, d = scan["y"], scan[channel], scan["d"]
    sigma = np.sqrt(np.maximum(counts, 1))

    def model(y_, N0, C, phi):
        return _fringe_model(y_, N0, C, phi, d)

    p0 = [max(counts.mean(), 1.0), 0.3, 0.0]
    bounds = ([0, 0, -np.pi], [np.inf, 1, np.pi])
    popt, pcov = curve_fit(model, y, counts, p0=p0, sigma=sigma,
                            absolute_sigma=True, bounds=bounds, maxfev=20000)
    perr = np.sqrt(np.diag(pcov))
    return popt, perr

# ===========================================================================
# Rozszerzenie: symulacja z rozrzutem kątowym (dywergencją) wiązki
# ===========================================================================

def sample_velocities_3d(n, v0, v_spread_rel, theta_div, rng):
    """
    Draw n atom velocities in 3D:
      - vz: longitudinal velocity (along z-axis) with Gaussian spread v_spread_rel
      - vx, vy: transverse velocities with spread theta_div * v0
    
    Returns arrays (vx, vy, vz) and the total speed v = sqrt(vx^2+vy^2+vz^2).
    """
    if v_spread_rel <= 0:
        vz = np.full(n, v0)
    else:
        sigma_v = v_spread_rel * v0
        vz = rng.normal(v0, sigma_v, size=n)
        vz = np.clip(vz, 0.05 * v0, None)
    
    # Transverse velocity spread from divergence angle
    sigma_perp = theta_div * v0
    vx = rng.normal(0, sigma_perp, size=n)
    vy = rng.normal(0, sigma_perp, size=n)
    
    # Total speed (for decay time and transit time)
    v = np.sqrt(vx**2 + vy**2 + vz**2)
    
    return vx, vy, vz, v

def simulate_grating3_scan_with_divergence_point(
    y, *, d, f_open, beam, L, C_max, g, phase_offset,
    gap_to_stopper, theta_div, n_atoms, rng, bragg_acceptance
):
    vx, vy, vz, v = sample_velocities_3d(n_atoms, beam.v0, beam.v_spread / beam.v0,
                                          theta_div, rng)
    tau = L / vz
    transit_time = 2 * L / v
    
    decay_time = rng.exponential(beam.lifetime, size=n_atoms)
    reaches_g3 = decay_time > transit_time
    n_lost_before = int(n_atoms - np.sum(reaches_g3))
    n_reach = int(np.sum(reaches_g3))
    
    if n_reach == 0:
        return 0, 0, n_lost_before, 0
    
    vx_r = vx[reaches_g3]
    vy_r = vy[reaches_g3]
    vz_r = vz[reaches_g3]
    tau_r = tau[reaches_g3]
    
    # --- NOWA FIZYKA: AKCEPTACJA BRAGGA ---
    # Obliczamy kąt padania każdego atomu (odchylenie od osi z)
    alpha = np.sqrt(vx_r**2 + vy_r**2) / vz_r
    
    # Atomy o zbyt dużym kącie tracą widoczność prążków, bo nie uginają się w siatce świetlnej.
    # Używamy modelu opadania Gaussa (szacując σ jako połowę szerokości Bragga)
    sigma_bragg = bragg_acceptance / 2.0
    c_atom = C_max * np.exp(- (alpha**2) / (2 * sigma_bragg**2))
    # ---------------------------------------
    
    # Faza grawitacyjna
    phi_g = 2 * np.pi / d * g * tau_r ** 2
    phi_total = phi_g + phase_offset
    
    # Faza interferencyjna (przekazujemy wektor c_atom zamiast stałego C_max!)
    theta = sample_fringe_phase_per_atom(phi_total, c_atom, rng)
    
    # Obliczenie trafień na fizyczną siatkę (maskę)
    y_phase = (2 * np.pi * y / d) % (2 * np.pi)
    rel = (theta - y_phase) % (2 * np.pi)
    crosses = rel < (2 * np.pi * f_open)
    
    n_stop = int(np.sum(~crosses))
    n_cand = int(np.sum(crosses))
    
    if gap_to_stopper > 0 and n_cand > 0:
        v_cross = v[reaches_g3][crosses]
        t_gap = gap_to_stopper / v_cross
        decay_gap = rng.exponential(beam.lifetime, size=n_cand)
        survive_gap = decay_gap > t_gap
        n_cross = int(np.sum(survive_gap))
        n_lost_gap = n_cand - n_cross
    else:
        n_cross = n_cand
        n_lost_gap = 0
    
    return n_cross, n_stop, n_lost_before, n_lost_gap

def simulate_grating3_scan_with_divergence(
    y_array, *, beam=PS_23S_MARIAZZI, L=0.2, f_open=0.5, C_max=0.8,
    g=psi.g_earth, phase_offset=0.0, gap_to_stopper=6e-3,
    theta_div=5.5e-3,
    bragg_acceptance=5.5e-3,  # Dodany domyślny kąt akceptacji z Mariazzi (2020)
    n_atoms_per_point=68000, seed=None
):
    rng = np.random.default_rng(seed)
    d = psi.grating_period(beam.lam_L)
    y_array = np.asarray(y_array, dtype=float)
    
    n_cross = np.zeros(len(y_array), dtype=int)
    n_stop = np.zeros(len(y_array), dtype=int)
    n_lost = np.zeros(len(y_array), dtype=int)
    
    for i, y in enumerate(y_array):
        nc, ns, nlb, nlg = simulate_grating3_scan_with_divergence_point(
            y, d=d, f_open=f_open, beam=beam, L=L, C_max=C_max, g=g,
            phase_offset=phase_offset, gap_to_stopper=gap_to_stopper,
            theta_div=theta_div, n_atoms=n_atoms_per_point, rng=rng,
            bragg_acceptance=bragg_acceptance # <--- Tutaj przekazujesz
        )
        n_cross[i], n_stop[i], n_lost[i] = nc, ns, nlb + nlg
    
    return dict(
        y=y_array, n_cross=n_cross, n_stop=n_stop, n_lost=n_lost,
        d=d, L=L, beam=beam, f_open=f_open, C_max=C_max, g=g,
        theta_div=theta_div, bragg_acceptance=bragg_acceptance, 
        n_atoms_per_point=n_atoms_per_point
    )