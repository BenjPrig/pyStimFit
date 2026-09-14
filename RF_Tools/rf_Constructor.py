#!/usr/bin/env python3
# -*- coding: utf-8 -*- #
# ============================================= #
# py RF reader
# B. Prigent
# 18-May-2026
# ============================================= #
'''
    Collection of low-level RF waveform parsers for various MRI vendors.
'''

import numpy as np
import logging 
import pandas as pd
from typing import Tuple, Optional
logger = logging.getLogger(__name__)

def _read_rf_varian(m_fname) -> np.ndarray:
    '''Reads a Varian/Agilent format RF pulse file.

    Args:
        m_fname (Path): Path to the .RF file.

    Returns:
        np.ndarray: Complex RF waveform array
    '''
    logger.info(f'Reading varian RF file: {m_fname}')
    rf_list = []
    
    with open(m_fname, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            vals = [float(x) for x in line.split()]
            if len(vals) < 3:
                continue
                
            phase, amp, reps = vals[0], vals[1], int(vals[2])
            
            if abs(phase - 180) < 0.01:
                rf_val = -amp
            elif abs(phase) < 0.001:
                rf_val = amp
            else:
                rf_val = amp * np.exp(1j * phase * np.pi / 180)
                
            rf_list.extend([rf_val] * reps)
    logger.info(f'{m_fname} converted successfully.')
            
    return np.array(rf_list, dtype=complex)


def _read_rf_siemens(m_fname) -> np.ndarray:
    '''Reads a Siemens format (.pta) RF pulse file.

    Args:
        m_fname (Path): Path to the .pta file.

    Returns:
        np.ndarray: Complex RF waveform array.
    '''
    logger.info(f'Reading siemens RF file: {m_fname}')
    df = pd.read_csv(filepath, sep='\s+', comment=';', header=None, skip_blank_lines=True)
    logger.info(f'{m_fname} converted successfully.')
    return np.array(rf_list, dtype=complex)


def _read_rf_ge(m_fname):
    '''Reads a GE format (.rho) RF pulse file.

    Args:
        m_fname (Path): Path to the .pta file.

    Returns:
        np.ndarray: Complex RF waveform array.
    '''
    logger.info(f'Reading GE RF file: {m_fname}')
    
    # '>' for Big-Endian, 'i2' for int16 (2 bytes)
    with open(m_fname, 'rb') as f:
        RF = np.fromfile(f, dtype='>i2')
        
    if len(RF) > 34:
        RF = RF[32:-2]
    else:
        RF = np.array([])
    logger.info(f'{m_fname} converted successfully.')
    return RF.astype(float)


def _interpolate_rf(m_RF, m_Nrf) -> np.ndarray:
    '''Linearly interpolates the RF signal to a target number of points N.

    Args:
        m_RF (np.ndarray): Original RF waveform array.
        m_Nrf (int): Desired number of points in the resampled array.

    Returns:
        np.ndarray: Resampled RF waveform array.
    '''
    if len(m_RF) <= 1:
        return np.zeros(m_Nrf)
        
    t1 = np.arange(len(m_RF))
    t2 = np.linspace(0, len(m_RF) - 1, m_Nrf)

    if np.iscomplexobj(m_RF):
        real_interp = np.interp(t2, t1, np.real(m_RF))
        imag_interp = np.interp(t2, t1, np.imag(m_RF))
        logger.info(f'Complex interpolation finished.')
        return real_interp + 1j * imag_interp
    else:
        logger.info(f'Real interpolation finished.')
        return np.interp(t2, t1, m_RF)
    
def _read_rf_simulation(m_RF) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    '''
    Reads a .txt simulation file
    Column 1 : Amplitude (B1)
    Column 3 : Phase (RF-Signal_Phase)

    Args:
        m_fname (Path): Path to the .pta file.

    Returns:
        np.ndarray: Complex RF waveform array
    '''
    
    logger.info(f"Reading .txt/tsv RF file: {m_RF.path}")
    
    df = pd.read_csv(m_RF.path, sep=r'\t', comment='#', engine='python')
    df.columns = [col.lower() for col in df.columns]
    
    logger.debug(f'Column of the df file {df.head}')
    
    # amp_col = next((c for c in df.columns if 'rf-signal(ch.0,1h,123.2mhz)' in c or 'amplitude' in c), None)
    amp_col = next((c for c in df.columns if 'rf-signal(ch.0,1h,123.2mhz)' in c or 'amplitude' in c), None)
    logger.debug(f"La colonne choisie pour l'amplitude est {amp_col}")
    # phase_col = next((c for c in df.columns if 'phase' in c), None)
    phase_col = next((c for c in df.columns if 'rf-signalphase(ch.0,1h,123.2mhz)' in c), None)

    if not amp_col or not phase_col:
        logger.error("Colonnes 'B1' (ou 'Amplitude') et/ou 'RF-Signal_Phase' (ou 'Phase') introuvables.")
        return np.array([], dtype=complex)
    
    amp = df[amp_col].to_numpy() # mT
    phase = df[phase_col].to_numpy()

    if m_RF.G == 0.0 :
            # J'ai ajouté d'autres noms potentiels classiques. 
            # Il faudra ajouter le vrai nom une fois que tu l'auras identifié.
            grad_candidates = ['zgradient', 'zgradient(gpa0)', 'gradient']
            grad_col = next((c for c in grad_candidates if c in df.columns), None)
            
            if grad_col is None:
                # Si la colonne n'est toujours pas trouvée, on log les colonnes existantes et on arrête proprement
                logger.error(f"Colonne de gradient introuvable. Colonnes disponibles dans le fichier : {list(df.columns)}")
                raise ValueError(f"Impossible de trouver le gradient dans le fichier {m_RF.path}. Vérifiez le nom des colonnes.")
                
            grad = df[grad_col].to_numpy()
    else:
        logger.info(f"Constant Gss found, taking the given value {m_RF.G}.")
        grad = np.ones(len(amp)) * m_RF.G
    
    rf_array = amp * np.exp(1j * np.radians(phase))
    
    logger.debug(f'Magnitude sum = {np.sum(amp)}, and {len(amp)} number of point')
    logger.debug(f'Phase sum = {np.sum(phase)}, and {len(phase)} number of point')
    logger.debug(f'Gradient sum = {np.sum(grad)}, and {len(grad)} number of point')
    return rf_array, grad

