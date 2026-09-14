#!/usr/bin/env python3
# -*- coding: utf-8 -*- #
# ============================================= #
# py pulse sim 
# B. Prigent
# 18-May-2026
# ============================================= #
'''

'''

import numpy as np
import logging
logger = logging.getLogger(__name__)

def pulse_sim(m_M0, m_z, m_RF, m_Gbg=0) -> np.ndarray:
    """
    Simulates magnetization behavior under an RF pulse using a vectorized Bloch simulator.
    """
    logger.info('Starting Bloch simulation loop.')

    # Rapport gyromagnétique pour 1H en rad / (s * Gauss)
    gamma = 2 * np.pi * 42.575e6 * 1e-4  # rad / (s * Gauss)
    Nt = len(m_RF.rf_waveform)
    
    if Nt == 0:
        logger.warning("Empty RF waveform provided to simulator. Returning initial magnetization.")
        return m_M0.copy()
    
    dt = m_RF.tau / Nt

    # Gestion de la forme du gradient
    if np.isscalar(m_RF.G):
        G_t = np.ones(Nt) * m_RF.G
    else:
        G_t = m_RF.G * 1e-1 # Conversion G / cm 
        if len(G_t) != Nt:
            raise ValueError("La longueur du tableau de gradient G doit correspondre à celle de rf_waveform.")

    # Pré-calcul des angles de nutation et phases RF instantanées
    theta_rf = gamma * np.abs(m_RF.rf_waveform) * dt
    phase_rf = np.angle(m_RF.rf_waveform) + np.radians(m_RF.phase)

    M = m_M0.astype(np.float64).copy()

    for i in range(Nt):
        # 1. Précession due au gradient
        phi_g = gamma * (G_t[i] + m_Gbg) * m_z * dt
        cphi, sphi = np.cos(phi_g), np.sin(phi_g)
        Mx, My = M[0, :].copy(), M[1, :].copy()
        M[0, :] = cphi * Mx + sphi * My
        M[1, :] = -sphi * Mx + cphi * My

        # 2. Nutation dans le repère de la RF
        theta = theta_rf[i]
        phi = phase_rf[i]
        
        c_p, s_p = np.cos(phi), np.sin(phi)
        ct, st = np.cos(theta), np.sin(theta)

        # Passage dans le repère de la RF
        Mx, My = M[0, :].copy(), M[1, :].copy()
        Mx_rot = c_p * Mx + s_p * My
        My_rot = -s_p * Mx + c_p * My

        # Nutation
        My_new = ct * My_rot + st * M[2, :]
        M[2, :] = -st * My_rot + ct * M[2, :]

        # Retour dans le repère du laboratoire
        M[0, :] = c_p * Mx_rot - s_p * My_new
        M[1, :] = s_p * Mx_rot + c_p * My_new

    if hasattr(m_RF, 'ref') and m_RF.ref > 0:
        logger.info(f"Applying refocusing gradient lobe (Fraction: {m_RF.ref})")
        
        G_integral = np.sum(G_t) * dt
        psi = -m_RF.ref / 2 * gamma * G_integral * m_z
        cpsi, spsi = np.cos(psi), np.sin(psi)
        
        Mx, My = M[0, :].copy(), M[1, :].copy()
        M[0, :] = cpsi * Mx + spsi * My
        M[1, :] = -spsi * Mx + cpsi * My

    logger.info('Bloch simulation completed successfully.')
    return M