## Setup

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional


class Material(ABC):
    """Base class for material properties."""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def specific_heat(self, temperature: float) -> float:
        """Calculate specific heat at given temperature."""
        pass
    
    @abstractmethod
    def thermal_conductivity(self, temperature: float) -> float:
        """Calculate thermal conductivity at given temperature."""
        pass
    
    @abstractmethod
    def density(self, temperature: float) -> float:
        """Calculate density at given temperature."""
        pass


class Alumina(Material):
    """Alumina solid material properties, taken from appendix E, D. Pérez-Gallego, J. Gonzalez-Ayala, A. Medina et al."""
    
    def __init__(self):
        super().__init__("Alumina")
        self.density_constant = 3550.0  # kg/m³
    
    def specific_heat(self, temperature: float) -> float:
        """Specific heat of alumina [J/kg*K]."""
        return -0.0022 * temperature**2 + 3.0637 * temperature + 65.5464
    
    def thermal_conductivity(self, temperature: float) -> float:
        """Thermal conductivity of alumina [W/m*K]."""
        return 0.0001 * temperature**2 - 0.1773 * temperature + 79.925
    
    def density(self, temperature: float) -> float:
        """Density of alumina [kg/m³]."""
        return self.density_constant


class Air(Material):
    """Air fluid material properties, taken from appendix E, D. Pérez-Gallego, J. Gonzalez-Ayala, A. Medina et al."""
    
    def __init__(self):
        super().__init__("Air")
    
    def specific_heat(self, temperature: float) -> float:
        """Specific heat of air [J/kg*K]."""
        return (1040.05 - 0.307614 * temperature + 
                0.000743982 * temperature**2 - 
                3.35122e-7 * temperature**3)
    
    def thermal_conductivity(self, temperature: float) -> float:
        """Thermal conductivity of air [W/m*K]."""
        return (0.00478072 + 0.0000772337 * temperature - 
                1.43807e-8 * temperature**2)
    
    def density(self, temperature: float) -> float:
        """Density of air [kg/m³]."""
        return (2.54676 - 0.00645271 * temperature + 
                6.84089e-6 * temperature**2 - 
                2.57974e-9 * temperature**3)
    
    def dynamic_viscosity(self, temperature: float) -> float:
        """Dynamic viscosity of air [Pa*s]."""
        return (4.62319e-6 + 5.07777e-8 * temperature - 
                1.21568e-11 * temperature**2)

class CalcHeatCoeffs:
    @staticmethod
    def calc_h(temperature: float, velocity: float, particle_diameter: float, porosity: float, fluid: Air) -> float:
        rho = fluid.density(temperature)
        mu = fluid.dynamic_viscosity(temperature)
        nu = mu / rho
        Re_p = velocity * porosity * particle_diameter / nu
        cp = fluid.specific_heat(temperature)
        k = fluid.thermal_conductivity(temperature)
        Pr = cp * mu / k
        Nu = 2.0 + 1.1 * Re_p**0.6 * Pr**(1/3)
        return k * Nu / particle_diameter

    @staticmethod
    def calc_biot(temperature: float, velocity: float, particle_diameter: float, porosity: float, solid: Alumina, fluid: Air) -> float:
        h = CalcHeatCoeffs.calc_h(temperature, velocity, particle_diameter, porosity, fluid)
        k_s = solid.thermal_conductivity(temperature)
        return h * particle_diameter / k_s


class Geometry:
    """Geometry parameters."""
    
    def __init__(self, bed_height: float, porosity: float, particle_diameter: float,
                 internal_radius: float):
        self.bed_height = bed_height
        self.porosity = porosity
        self.particle_diameter = particle_diameter
        self.internal_radius = internal_radius
        self.specific_surface_area = 6 * (1 - porosity) / particle_diameter

class ImplicitEuler:
    @staticmethod
    def TDMA(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
        """
        Solve tridiagonal system with subdiag a, diag b, superdiag c, RHS d.
        Arrays are length N; a[0] and c[-1] ignored.
        """
        n = len(b)
        cp = np.empty(n - 1)
        dp = np.empty(n)

        # guard
        if abs(b[0]) < 1e-14:
            raise ZeroDivisionError("TDMA: b[0] is zero.")

        cp[0] = c[0] / b[0]
        dp[0] = d[0] / b[0]
        for i in range(1, n - 1):
            denom = b[i] - a[i] * cp[i - 1]
            if abs(denom) < 1e-14:
                raise ZeroDivisionError(f"TDMA: zero pivot at i={i}.")
            cp[i] = c[i] / denom
            dp[i] = (d[i] - a[i] * dp[i - 1]) / denom

        denom_last = b[n - 1] - a[n - 1] * cp[n - 2]
        if abs(denom_last) < 1e-14:
            raise ZeroDivisionError("TDMA: zero pivot at last row.")
        dp[n - 1] = (d[n - 1] - a[n - 1] * dp[n - 2]) / denom_last

        x = np.empty(n)
        x[-1] = dp[-1]
        for i in range(n - 2, -1, -1):
            x[i] = dp[i] - cp[i] * x[i + 1]
        return x

    # def assemble_TDMA_Tf(
    #     self,
    #     Tf_old: np.ndarray,
    #     Ts_old: np.ndarray,
    #     dt: float,
    #     dz: float,
    #     u: float,
    #     eps: float,
    #     rhof: float,
    #     cpf: float,
    #     rhos: float,
    #     cps: float,
    #     ha: float,
    #     Tf_in_next: float,
    # ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    #     """
    #     Build tridiagonal system for T_f^{t+dt} after eliminating T_s^{t+dt} locally.
    #     Assumes u > 0 (first-order upwind) -> lower bidiagonal (c[:] = 0).
    #     """
    #     N = Tf_old.size

    #     # Effective volumetric heat capacities
    #     ef = eps * rhof * cpf                 # fluid
    #     es = (1.0 - eps) * rhos * cps         # solid

    #     A = ef / dt
    #     S  = es / dt
    #     H     = ha
    #     K     = H * S / (S + H)         # = HS/(S+H) with S = es/dt
    #     beta   = ef * u / dz                   # >= 0 for u>0
    #     beta_max = max(beta, 0.0)
    #     beta_min = min(beta, 0.0)
    #     alpha = A + (beta_max - beta_min) + K

    #     a = np.zeros(N)
    #     b = np.zeros(N)
    #     c = np.zeros(N)
    #     d = np.zeros(N)

    #     # Inlet Dirichlet at t+dt

    #     if u>= 0:
    #         b[0] = 1.0
    #         d[0] = Tf_in_next
    #         c[0] = 0.0
    #         a[0] = 0.0
    #     else:
    #         b[-1] = 1.0
    #         d[-1] = Tf_in_next
    #         c[-1] = 0.0
    #         a[-1] = 0.0

    #     # Interior & outlet rows (upwind, u>0)
    #     for i in range(1, N):
    #         a[i] = -beta
    #         b[i] = alpha
    #         c[i] = 0.0
    #         d[i] = A * Tf_old[i] + K * Ts_old[i]

    #     return a, b, c, d

    def assemble_TDMA_Tf(
        self,
        Tf_old: np.ndarray,
        Ts_old: np.ndarray,
        dt: float,
        dz: float,
        u: float,
        eps: float,
        rhof: float,
        cpf: float,
        rhos: float,
        cps: float,
        ha: float,
        Tf_in_next: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Build tridiagonal system for T_f^{t+dt} after eliminating T_s^{t+dt} locally.
        Sign-aware upwind advection:
        - if u >= 0: inlet at i=0; use lower bidiagonal (c[:] mostly 0)
        - if u <  0: inlet at i=N-1; use upper bidiagonal (a[:] mostly 0)
        """
        N = Tf_old.size

        # Effective volumetric heat capacities
        ef = eps * rhof * cpf            # fluid
        es = (1.0 - eps) * rhos * cps    # solid

        A = ef / dt
        S = es / dt
        H = ha
        K = H * S / (S + H)

        beta = ef * u / dz
        beta_p = max(beta, 0.0)          # ≥ 0
        beta_m = min(beta, 0.0)          # ≤ 0

        a = np.zeros(N)                  # subdiagonal
        b = np.zeros(N)                  # diagonal
        c = np.zeros(N)                  # superdiagonal
        d = np.zeros(N)                  # RHS

        # Inlet Dirichlet at the upwind side
        if u >= 0.0:
            inlet = 0
            b[inlet] = 1.0
            d[inlet] = Tf_in_next

            # interior rows i = 1..N-1
            for i in range(1, N):
                a[i] = -beta_p                  # = -beta
                b[i] = A + (beta_p - beta_m) + K  # A + |beta| + K
                c[i] = beta_m                   # = 0 for u>=0
                d[i] = A * Tf_old[i] + K * Ts_old[i]

        else:
            inlet = N - 1
            b[inlet] = 1.0
            d[inlet] = Tf_in_next

            # interior rows i = 0..N-2
            for i in range(0, N-1):
                a[i] = -beta_p                  # = 0 for u<0
                b[i] = A + (beta_p - beta_m) + K  # A + |beta| + K
                c[i] = beta_m                   # < 0 for u<0 (upwind)
                d[i] = A * Tf_old[i] + K * Ts_old[i]

        return a, b, c, d


    @staticmethod
    def recover_Ts(
        Tf_new: np.ndarray,
        Ts_old: np.ndarray,
        dt: float,
        eps: float,
        rhos: float,
        cps: float,
        ha: float,
    ) -> np.ndarray:
        es   = (1.0 - eps) * rhos * cps
        S = es / dt
        H    = ha
        return (H * Tf_new + S * Ts_old) / (S + H)


