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
    gamma = 2 * np.pi * 42.575e6 * 1e-4
    Nt = len(m_RF.rf_waveform)
    dt =  m_RF.tau / Nt
    
    if Nt == 0:
        logger.warning("Empty RF waveform provided to simulator. Returning initial magnetization.")
        return m_M0.copy()
    
    if np.isscalar(m_RF.G):
        G_t = np.ones(Nt) * m_RF.G
    else:
        G_t = m_RF.G * 1e-1
        if len(G_t) != Nt:
            raise ValueError("La longueur du tableau de gradient G doit correspondre à celle de rf_waveform.")
    
    
    # rf_gauss = m_RF.rf_waveform 

    # 5. Calcul des angles de rotation (Nutation et Précession)
    theta_rf = gamma * np.abs(m_RF.rf_waveform) * dt
    phase_rf = np.angle(m_RF.rf_waveform)

    # Phase spatiale accumulée par le gradient (G_t en Gauss/cm, m_z en cm)
    theta_g = gamma * G_t[:, None] * m_z[None, :] * dt

    # Paramètres de phase globale de l'impulsion RF
    rf_phase_rad = np.radians(m_RF.phase)
    cp_rf = np.cos(rf_phase_rad)
    sp_rf = np.sin(rf_phase_rad)
    
    # theta_rf = gamma * np.real(m_RF.rf_waveform) * m_RF.tau
    
    M = m_M0.astype(np.float64).copy()

    # Boucle temporelle (Hard-pulse approximation)
    for i in range(Nt):
        # 1. Precession : Le calcul du déphasage se fait maintenant ICI à chaque pas
        phi = gamma * (G_t[i] + m_Gbg) * m_z * dt
        cphi = np.cos(phi)
        sphi = np.sin(phi)

        Mx, My = M[0, :].copy(), M[1, :].copy()
        M[0, :] = cphi * Mx + sphi * My
        M[1, :] = -sphi * Mx + cphi * My
        
        # 2. Nutation (inchangé)
        theta = theta_rf[i]
        ct = np.cos(theta)
        st = np.sin(theta)
        
        if m_RF.phase != 0.0:
            Mx, My = M[0, :].copy(), M[1, :].copy()
            M[0, :] = cp_rf * Mx + sp_rf * My
            M[1, :] = -sp_rf * Mx + cp_rf * My
            
            My, Mz = M[1, :].copy(), M[2, :].copy()
            M[1, :] = ct * My + st * Mz
            M[2, :] = -st * My + ct * Mz
            
            Mx, My = M[0, :].copy(), M[1, :].copy()
            M[0, :] = cp_rf * Mx - sp_rf * My
            M[1, :] = sp_rf * Mx + cp_rf * My
        else:
            My, Mz = M[1, :].copy(), M[2, :].copy()
            M[1, :] = ct * My + st * Mz
            M[2, :] = -st * My + ct * Mz

    # =========================================================================
    # CORRECTION 3 : AJOUT DU LOBE DE REFOCALISATION (Rephase magnetization)
    # =========================================================================
    if hasattr(m_RF, 'ref') and m_RF.ref > 0:
        logger.info(f"Applying refocusing gradient lobe (Fraction: {m_RF.ref})")
        
        # L'aire du gradient de sélection est l'intégrale de G(t)
        G_integral = np.sum(G_t) * dt
        
        # psi remplace G * tau par G_integral (équivalent strict si G est constant)
        psi = -m_RF.ref / 2 * gamma * G_integral * m_z
        cpsi = np.cos(psi)
        spsi = np.sin(psi)
        
        Mx, My = M[0, :].copy(), M[1, :].copy()
        M[0, :] = cpsi * Mx + spsi * My
        M[1, :] = -spsi * Mx + cpsi * My

    logger.info('Bloch simulation completed successfully.')

    return M