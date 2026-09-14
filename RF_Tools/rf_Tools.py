#!/usr/bin/env python3
# -*- coding: utf-8 -*- #
# ============================================= #
# py RF tools
# B. Prigent
# 18-May-2026
# ============================================= #
'''
    Collection of RF manipulation adaptated from StimFit toolbox for python.
'''
import logging
import numpy as np
from typing import Tuple, Optional
from pyStimFit.RF_Tools.class_rf import RfWaveform
from pyStimFit.RF_Tools.pulse_sim import pulse_sim
from baseLogger import setup_logging
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
import pyStimFit.RF_Tools.rf_Constructor as rfCons


logger = logging.getLogger(__name__)

def read_RF(m_RF)-> RfWaveform:
    '''
    Opens an RF pulse file (via GUI if no path provided), parses it, and interpolates it.

    Args:
        m_RF (Object): Object of the dataclass RfWaveform
    
    Returns:
        - RfWaveform object containing the original data.
    '''
    if not m_RF.path:
        logger.debug('No file provided, opening GUI file selector.')
        root = tk.Tk()
        root.withdraw()  
        
        ftypes = [
            ('Varian RF file', '*.RF'),
            ('Siemens RF file', '*.pta'),
            ('GE RF file', '*.rho'),
            ('Text/Simulation file', '*.txt'),
            ('All files', '*.*')
        ]
        selected_file = filedialog.askopenfilename(filetypes=ftypes)
        if not selected_file:
            logger.warning('No file selected by the user.')
            return None, None
        m_RF.path = selected_file
    
    m_RF.path = Path(m_RF.path)
    logger.info(f'Reading RF file from: {m_RF.path}')
    ext = m_RF.path.suffix.lower()
    logger.debug(f'Detected file extension: {ext}')

    if ext == '.rf':
        RF = rfCons._read_rf_varian(str(m_RF.path))
    elif ext == '.pta':
        RF = rfCons._read_rf_siemens(str(m_RF.path))
    elif ext == '.rho':
        RF = rfCons._read_rf_ge(str(m_RF.path))
    elif ext == '.txt':
        RF, grad = rfCons._read_rf_simulation(m_RF)
    else:
        raise ValueError(f"RF file extension '{ext}' is not supported.")
    
    # !NOTE: Keep waveform complex if the downstream simulation code supports it.
    # If pulse_sim strictly requires real arrays, uncomment the block below:
    # if np.iscomplexobj(RF) and np.any(np.iscomplex(RF)):
    #     logger.warning('Complex RF waveforms not yet supported by simulation. Dropping imaginary part.')
    #     RF = np.real(RF)
    
    # Generate interpolated version

    rf_object = RfWaveform(
        path=m_RF.path,
        rf_waveform=RF.astype(complex),
        G=grad
    )
    return rf_object 

def scale_RF(m_RF)-> RfWaveform:
    '''
    Scales the RF waveform amplitude to achieve the prescribed nominal flip angle.

    Calculates the current flip angle area based on the pulse duration (tau) 
    and gyromagnetic ratio, then applies a scaling factor to convert the arbitrary 
    waveform units into Gauss.

    Args:
        m_RF (RfWaveform): The RF waveform object containing the physical parameters 
                           (tau, angle) and the raw pulse profile.

    Returns:
        RfWaveform: The sub-structure with the scaled waveform (in Gauss).
    '''
    logger.info('Starting RF amplitude scaling.')
    gamma = 2 * np.pi * 42.575e6 * 1e-4  # rad / (s * Gauss)
    
    current_area = np.abs(np.sum(m_RF.rf_waveform))
    dt = m_RF.tau / len(m_RF.rf_waveform)
    alphaC = gamma * dt * current_area
    alphaC = np.degrees(alphaC)
    
    # 2. Facteur d'échelle = Angle voulu / Angle actuel
    if alphaC == 0:
        scaling_factor = 1.0
        logger.warning("RF waveform integral is zero or pulse duration (tau) is null. Forcing scaling factor to 1.0.")
    else:
        scaling_factor = m_RF.angle / alphaC
        
    m_RF.rf_waveform = m_RF.rf_waveform * scaling_factor
    
    logger.debug(f'Gyromagnetic ratio (gamma): {gamma} rad/(s*G)')
    logger.debug(f'Target nominal angle: {m_RF.angle}°')
    logger.info(f"RF Waveform successfully scaled by a factor of {scaling_factor:.6f} (Units: Gauss).")
    
    return m_RF

def calc_FlipRF(m_RF, m_Dz, m_Nz, m_Gbg) -> RfWaveform:
    '''Computes the spatial flip angle distribution (slice profile) using the Bloch simulator.

    Args:
        m_RF (RfWaveform): RF pulse waveform structure.
        m_Dz (Tuple[float, float]): Spatial bounds (z_min, z_max) along the slice axis in cm.
        m_Nz (int): Number of spatial points to sample within m_Dz.
        m_Gbg (float): Background field gradient amplitude (G/cm). Defaults to 0.0.

    Returns:
        RfWaveform: The updated RF object containing the populated .alpha distribution field.
    '''
    logger.debug('Magnetization initialization (Mz = 1)')
    z_axis = np.linspace(m_Dz[0], m_Dz[1], m_Nz)
    
    # 1. On initialise TOUJOURS sur l'axe Z pour calculer le profil de bascule pur
    #    (Que ce soit une excitation 90° ou une refocalisation 180°)
    M0 = np.zeros((3, len(z_axis)))
    M0[2, :] = 1.0 
    
    # 2. Simulation de Bloch
    M_final = pulse_sim(M0, z_axis, m_RF, m_Gbg)
    
    # 3. Extraction de l'angle 
    Mx = M_final[0, :]
    My = M_final[1, :]
    Mz = M_final[2, :]
    
    # La magnitude transverse M_perp est immunisée contre le déphasage (phase wind-up)
    # causé par le gradient de sélection de coupe pendant l'impulsion.
    M_perp = np.sqrt(Mx**2 + My**2)
    
    # arctan2 gère nativement la bascule géométrique de [0, 180°]
    alpha_rad = np.arctan2(M_perp, Mz)
    m_RF.alpha = np.degrees(alpha_rad)

    logger.info(f"Spatial flip angle distribution (.alpha) successfully evaluated.")
    return m_RF

# def calc_FlipRF(m_RF, m_Dz, m_Nz, m_Gbg) -> RfWaveform:
#     '''Computes the spatial flip angle distribution (slice profile) using the Bloch simulator.

#     Uses a low-flip-angle scaling approximation trick to avoid arccos truncation 
#     and numerical instabilities.

#     Args:
#         m_RF (RfWaveform): RF pulse waveform structure.
#         m_Dz (Tuple[float, float]): Spatial bounds (z_min, z_max) along the slice axis in cm.
#         m_Nz (int): Number of spatial points to sample within m_Dz.
#         m_Gbg (float): Background field gradient amplitude (G/cm). Defaults to 0.0.

#     Returns:
#         RfWaveform: The updated RF object containing the populated .alpha distribution field.
#     '''
#     logger.debug('Magnetization initialization')
#     z_axis = np.linspace(m_Dz[0], m_Dz[1], m_Nz)
    
#     # 1. Aimantation initiale (Mz = 1.0 partout)
#     M0 = np.zeros((3, len(z_axis)))
#     M0[2, :] = 1.0 
    
#     rf_original = m_RF.rf_waveform.copy()
#     m_RF.rf_waveform = 1e-4 * rf_original
#     M_final = pulse_sim(M0, z_axis, m_RF, m_Gbg)
    
#     m_RF.rf_waveform = rf_original
#     mz_clipped = np.clip(M_final[2, :], -1.0, 1.0)
#     m_RF.alpha = 1e4 * np.degrees(np.arccos(mz_clipped))

#     logger.info("Spatial flip angle distribution (.alpha) successfully evaluated in degrees.")
#     return m_RF

def get_RF(m_RF, m_Nrf, m_Dz, m_Nz):
    '''Populates the RF waveform structure by loading, scaling, and computing its flip angle profile.

    Args:
        m_RF (RfWaveform): Initial RF object holding at least the file path, tau, and target angle.
        m_Nrf (int): Number of points for the interpolated waveform profile (ignored for VERSE).
        m_Dz (Tuple[float, float]): Spatial range (z_min, z_max) in cm for Bloch simulation.
        m_Nz (int): Number of spatial points along the z-axis.

    Returns:
        Optional[RfWaveform]: Fully populated RfWaveform object, or None if file loading failed.
    '''
    tau_cache = m_RF.tau
    angle_cache = m_RF.angle
    G_cache = m_RF.G
    ref_cache = m_RF.ref
    phase_cache = m_RF.phase
    loaded_rf = read_RF(m_RF)
    
    if loaded_rf is None:
        logger.error("Failed to load the RF waveform file.")
        return None
    
    loaded_rf.tau = tau_cache
    loaded_rf.angle = angle_cache
    loaded_rf.ref = ref_cache
    loaded_rf.phase = phase_cache
    
    if G_cache != 0.0:
        loaded_rf.G = G_cache

    is_dynamic_grad = isinstance(loaded_rf.G, (list, np.ndarray)) and len(loaded_rf.G) > 1

    if is_dynamic_grad:
        logger.info("Impulsion VERSE détectée. Interpolation conjointe RF + Gradient.")
        if len(loaded_rf.rf_waveform) != m_Nrf:
            t_old = np.linspace(0, 1, len(loaded_rf.rf_waveform))
            t_new = np.linspace(0, 1, m_Nrf)
            
            # Interpolation complexe pour le B1
            r_interp = np.interp(t_new, t_old, np.real(loaded_rf.rf_waveform))
            i_interp = np.interp(t_new, t_old, np.imag(loaded_rf.rf_waveform))
            loaded_rf.rf_waveform = r_interp + 1j * i_interp
            
            # Interpolation pour le gradient
            loaded_rf.G = np.interp(t_new, t_old, loaded_rf.G)
        else:
            interp_waveform = rfCons._interpolate_rf(loaded_rf.rf_waveform, m_Nrf)
            if interp_waveform is not None:
                loaded_rf.rf_waveform = interp_waveform
            
    # Application de l'échelle pour la conversion en Gauss
    logging.debug(f'Sum of the real and imaginary part of rf {(np.sum(loaded_rf.rf_waveform))}')
    loaded_rf_scaled = scale_RF(loaded_rf)
    
    # Évaluation du profil de coupe spatial
    final_rf = calc_FlipRF(loaded_rf_scaled, m_Dz, m_Nz, m_Gbg=0.0)

    return final_rf