"""
positronium_interferometer.py
==============================

Python implementation of the physics in:

    M.K. Oberthaler, "Anti-matter wave interferometry with positronium",
    Nucl. Instr. and Meth. in Phys. Res. B 192 (2002) 129-134.

The module implements, in order of appearance in the paper:

    * Eq. (1)  Mach-Zehnder interference pattern  N_out(dx)
    * Eq. (2)  Gravitational phase shift          phi_g
    * Eq. (3)  Gravitational sensitivity           S
    * Eq. (4)  Light-shift (AC Stark) potential    U(r)
    * Eq. (5)  Talbot length                      L_T
    * Eq. (6)  Bragg diffraction efficiency        I_B
    * Eq. (7)  Required laser intensity for a Bragg "mirror"
    * Eq. (8)/(9) Spontaneous emission probability p_s

plus the beam kinematics (de Broglie wavelength, diffraction angle) and the
saturation-intensity relation used to build Table 1 of the paper.

A note on faithfulness
-----------------------
The PDF that this module is based on is an old (2002) two-column scan and
several equations were corrupted by OCR (Greek letters, subscripts and
overbars are frequently dropped or misread, e.g. "lambda" <-> "k", or
"hbar" <-> "h"). Wherever a formula could be *numerically validated*
against the worked example in Table 1 of the paper, this has been done
(see ``demo.py`` / ``print_table_1()``) and the implementation below
matches the paper to the printed precision:

    * de Broglie wavelength / diffraction angle theta   -> matches Table 1
    * Talbot length L_T                                 -> matches Table 1
    * saturation intensity I_s (from the Einstein A coeff.) -> matches Table 1

For Eqs. (6)-(9) (Bragg efficiency, required laser power, spontaneous
emission) the OCR damage was severe enough that the exact numerical
prefactors cannot be fully reconstructed with confidence. The implementation
below uses the standard, dimensionally-consistent atom-optics form of these
relations (the same author used essentially this formalism in the earlier
reference Oberthaler et al., Phys. Rev. A 60, 456 (1999), cited as Ref. [10]
of this paper). Using it reproduces the *order of magnitude* of the "necessary
laser power" column of Table 1, but not the figure to the last digit -- so
the literal Table-1 values are also kept as reference data in
``TABLE_1`` for comparison rather than re-derived blindly.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

# ---------------------------------------------------------------------------
# Physical constants (CODATA / SI units)
# ---------------------------------------------------------------------------
h = 6.62607015e-34          # Planck constant [J s]
hbar = h / (2 * np.pi)      # reduced Planck constant [J s]
c = 2.99792458e8            # speed of light [m/s]
m_e = 9.1093837015e-31      # electron mass [kg]
eV = 1.602176634e-19        # 1 eV in Joules [J]
g_earth = 9.80665           # standard gravity [m/s^2]

# Positronium (Ps = e- e+ bound state) mass. The ~6.8 eV binding energy is
# negligible compared to 2 * m_e c^2 = 1.022 MeV, so m_Ps = 2 m_e to very
# high accuracy for all kinematic purposes here.
m_Ps = 2 * m_e


# ---------------------------------------------------------------------------
# Beam kinematics
# ---------------------------------------------------------------------------
def ps_velocity(E_kin_eV: float) -> float:
    """Non-relativistic velocity of a Ps atom of kinetic energy E_kin_eV [eV]."""
    E = E_kin_eV * eV
    return np.sqrt(2 * E / m_Ps)


def de_broglie_wavelength(E_kin_eV: float) -> float:
    """De Broglie wavelength lambda_dB = h / (m_Ps v) of a Ps atom [m]."""
    v = ps_velocity(E_kin_eV)
    return h / (m_Ps * v)


# ---------------------------------------------------------------------------
# Standing-light-wave grating geometry (Sec. 3)
# ---------------------------------------------------------------------------
def grating_period(lam_L: float) -> float:
    """Period of the light-shift potential created by a standing light wave
    of wavelength lam_L:   d = lam_L / 2   [m]."""
    return lam_L / 2


def diffraction_angle(lam_L: float, E_kin_eV: float) -> float:
    """First-order Bragg diffraction angle (Sec. 3):
         theta = 2 * lambda_dB / lambda_L     [rad]
    """
    lam_dB = de_broglie_wavelength(E_kin_eV)
    return 2 * lam_dB / lam_L


def talbot_length(lam_L: float, E_kin_eV: float) -> float:
    """Talbot length, Eq. (5):
         L_T = 2 d^2 / lambda_dB     [m]
    sets the natural interaction length for Bragg-regime diffraction.
    """
    d = grating_period(lam_L)
    lam_dB = de_broglie_wavelength(E_kin_eV)
    return 2 * d ** 2 / lam_dB


# ---------------------------------------------------------------------------
# Light - atom (light - Ps) interaction
# ---------------------------------------------------------------------------
def saturation_intensity(Gamma: float, lam_L: float) -> float:
    """Two-level saturation intensity from the Einstein A coefficient:
         I_s = pi h c Gamma / (3 lambda_L^3)     [W/m^2]
    """
    return np.pi * h * c * Gamma / (3 * lam_L ** 3)


def light_shift_potential(I: float, Gamma: float, Is: float, detuning: float) -> float:
    """Far-detuned light-shift (AC Stark) potential, Eq. (4):
         U = hbar * Gamma^2 / (8 * Is) * I / detuning      [J]
    valid for |detuning| >> Gamma and |detuning| >> Rabi frequency.
    `detuning` and `Gamma` must use the same angular convention (rad/s).
    """
    return hbar * Gamma ** 2 / (8 * Is) * I / detuning


def bragg_efficiency(U_max: float, tau: float, order: int = 1) -> float:
    """n-th order Bragg diffraction efficiency for a phase grating of
    modulation depth U_max acting for an interaction time tau, Eq. (6):
         I_B = sin^2( n * U_max * tau / (2 hbar) )
    I_B = 0.5 gives a 50/50 beamsplitter, I_B = 1 a perfect mirror.
    (Standard atom-optics Bragg-scattering form; see module docstring.)
    """
    return np.sin(order * U_max * tau / (2 * hbar)) ** 2


def U_max_for_mirror(tau: float, order: int = 1) -> float:
    """Modulation depth U_max needed for a perfect Bragg mirror
    (I_B = 1), inverting bragg_efficiency()."""
    return np.pi * hbar / (2 * order * tau)


def required_intensity_for_mirror(Gamma: float, Is: float, detuning: float,
                                   tau: float, order: int = 1) -> float:
    """Laser intensity needed to realise a Bragg mirror (I_B = 1) for a
    given detuning and interaction time, combining the mirror condition
    with the light-shift potential of Eq. (4):
         I_in = 8 Is * detuning * U_max / (hbar * Gamma^2)
    """
    U_max = U_max_for_mirror(tau, order)
    return 8 * Is * detuning * U_max / (hbar * Gamma ** 2)


def spontaneous_emission_probability(Gamma: float, detuning: float) -> float:
    """Simplified spontaneous emission probability during a Bragg-mirror
    pulse, Eq. (9):
         p_s = 2 pi Gamma / detuning
    """
    return 2 * np.pi * Gamma / detuning


def detuning_for_target_ps(Gamma: float, p_s_target: float) -> float:
    """Invert Eq. (9): detuning needed to keep spontaneous emission
    probability at p_s_target."""
    return 2 * np.pi * Gamma / p_s_target


# ---------------------------------------------------------------------------
# Mach-Zehnder interferometer (Sec. 2)
# ---------------------------------------------------------------------------
def interference_pattern(dx, N0: float, C: float, d: float, phi: float = 0.0):
    """Mach-Zehnder fringe pattern, Eq. (1):
         N_out(dx) = N0 * [ 1 + C * cos(2 pi dx / d + phi) ]
    `dx` may be a scalar or numpy array (the third-grating displacement).
    """
    dx = np.asarray(dx)
    return N0 * (1 + C * np.cos(2 * np.pi * dx / d + phi))


def gravity_phase_shift(d: float, g: float, tau: float = None,
                         L: float = None, v: float = None) -> float:
    """Gravitational phase shift, Eq. (2):
         phi_g = 2 pi / d * g * tau^2
    where tau = L / v is the time spent inside the interferometer.
    Either pass `tau` directly, or both `L` and `v`.
    """
    if tau is None:
        if L is None or v is None:
            raise ValueError("Provide either tau, or both L and v.")
        tau = L / v
    return 2 * np.pi / d * g * tau ** 2


def sensitivity(C: float, N0: float, d: float, tau: float) -> float:
    """Minimum detectable acceleration (1 std. dev.), Eq. (3):
         S = (1 / (C sqrt(N0))) * (d / 2pi) * (1 / tau^2)
    `N0` is the *total number of detected particles* for the measurement
    (i.e. count rate x measurement time), `tau` the interaction time.
    Returns S in m/s^2.
    """
    return (1.0 / (C * np.sqrt(N0))) * (d / (2 * np.pi)) * (1.0 / tau ** 2)


# ---------------------------------------------------------------------------
# Table 1 of the paper: ortho-positronium states used for the interferometer
# ---------------------------------------------------------------------------
@dataclass
class PsState:
    name: str
    lifetime: float          # [s]
    lam_L: float              # driving optical transition wavelength [m]
    Gamma: float               # Einstein A coefficient [s^-1]


PS_STATES = {
    "1s": PsState(name="1s o-Ps (1s-2p)", lifetime=142e-9,
                  lam_L=243.1e-9, Gamma=3.17e8),
    "2s": PsState(name="2s o-Ps (2s-3p)", lifetime=1.1e-6,
                  lam_L=1312.5e-9, Gamma=1.13e7),
}

E_KIN_EV = 10.0  # kinetic energy of the Ps beam assumed throughout Table 1

# Literal values quoted in Table 1 of the paper, kept here as reference
# data for comparison with the values computed by this module.
TABLE_1_PAPER = {
    "1s": dict(theta_mrad=2.3, L_T_mm=0.1, Is_mWcm2=460, power=42.0,
               power_unit="W"),
    "2s": dict(theta_mrad=0.42, L_T_mm=3.1, Is_mWcm2=0.1, power=260e-3,
               power_unit="W"),
}
