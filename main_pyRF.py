#!/usr/bin/env python3
# -*- coding: utf-8 -*- #
# ============================================= #
# py test RF
# B. Prigent
# 18-May-2026
# ============================================= #
'''
    Main validation and testing script for the Python StimFit RF module.
    Loads excitation and refocusing pulses, runs Bloch simulations, and plots profiles.
'''

import os
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import time
import argparse

import pyStimFit.RF_Tools.class_rf as rf
import pyStimFit.RF_Tools.rf_Tools as tool
from baseLogger import setup_logging  # Assumed available in your environment

logger = logging.getLogger(__name__)

def export_test_data_to_tsv(excitation_rf: rf.RfWaveform, z_bounds: tuple,output_dir: str = "./results/") -> None:
    """Exports resampled RF waveform data and spatial slice profiles to TSV files.

    Args:
        excitation_rf (rf.RfWaveform): The fully populated RF waveform object.
        output_dir (str): Directory where the TSV files will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = Path(output_dir)
    
    # 1. EXPORT RESAMPLED RF WAVEFORM (Temporal Dimension: Nrf)
    df_rf = pd.DataFrame({
        'Time_Index': np.arange(1, len(excitation_rf.rf_waveform) + 1),
        'RF_Real_G': np.real(excitation_rf.rf_waveform),
        'RF_Imag_G': np.imag(excitation_rf.rf_waveform),
        'RF_Magnitude_G': np.abs(excitation_rf.rf_waveform)
    })
    rf_path = output_path / "scaled_rf_waveform.tsv"
    df_rf.to_csv(rf_path, sep='\t', index=False)
    logger.info(f"RF Waveform profile successfully exported to: {rf_path}")
    
    # 2. EXPORT SLICE PROFILE (Spatial Dimension: Nz)
    # Recreate the exact spatial axis used inside calc_FlipRF
    # (Assuming a standard 100-point grid or matching the length of alpha)
    nz_points = len(excitation_rf.alpha)
    z_axis = np.linspace(z_bounds[0], z_bounds[1], nz_points)  # Dummy range placeholder matching test bounds
    
    df_slice = pd.DataFrame({
        'Z_Position_cm': z_axis,
        'Flip_Angle_Deg': excitation_rf.alpha
    })
    slice_path = output_path / "simulated_slice_profile.tsv"
    df_slice.to_csv(slice_path, sep='\t', index=False)
    logger.info(f"Slice profile distribution successfully exported to: {slice_path}")

def plot_rf_and_profile_dynamic(excitation_rf: rf.RfWaveform, refocus_rf: rf.RfWaveform, z_bounds: tuple, nz: int, save_path: Path) -> None:
    """Generates a matplotlib window to visually check the RF shapes, Bloch flip angle profiles, and Normalized Mt."""
    z_axis = np.linspace(z_bounds[0], z_bounds[1], nz)
    time_exc = np.linspace(0, excitation_rf.tau * 1e3, len(excitation_rf.rf_waveform)) # ms
    time_ref = np.linspace(0, refocus_rf.tau * 1e3, len(refocus_rf.rf_waveform)) # ms

    # Création d'une grille 2 lignes x 3 colonnes
    fig, axs = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle("StimFit Python RF Module - Validation Plots", fontsize=14, fontweight='bold')

    # --- COLONNE 1 : RF Waveforms ---
    axs[0, 0].plot(time_exc, np.real(excitation_rf.rf_waveform), label='Real (I)', color='tab:blue')
    axs[0, 0].plot(time_exc, np.imag(excitation_rf.rf_waveform), label='Imag (Q)', color='tab:orange', linestyle='--')
    axs[0, 0].plot(time_exc, np.abs(excitation_rf.rf_waveform), label='Magnitude', color='black', alpha=0.5)
    axs[0, 0].set_title(f"Excitation Waveform ({excitation_rf.angle}°)")
    axs[0, 0].set_xlabel("Time (ms)")
    axs[0, 0].set_ylabel("Amplitude")
    axs[0, 0].grid(True)
    axs[0, 0].legend()

    axs[1, 0].plot(time_ref, np.real(refocus_rf.rf_waveform), label='Real (I)', color='tab:red')
    axs[1, 0].plot(time_ref, np.imag(refocus_rf.rf_waveform), label='Imag (Q)', color='tab:orange', linestyle='--')
    axs[1, 0].plot(time_ref, np.abs(refocus_rf.rf_waveform), label='Magnitude', color='black', alpha=0.5)
    axs[1, 0].set_title(f"Refocusing Waveform ({refocus_rf.angle}°)")
    axs[1, 0].set_xlabel("Time (ms)")
    axs[1, 0].set_ylabel("Amplitude")
    axs[1, 0].grid(True)
    axs[1, 0].legend()

        # --- COLONNE 2 : Gradient ---
    axs[0, 1].plot(time_exc, np.real(excitation_rf.G), label='Grad Exc', color='tab:blue')
    axs[0, 1].set_title(f"Gradient")
    axs[0, 1].set_xlabel("Time (ms)")
    axs[0, 1].set_ylabel("Amplitude")
    axs[0, 1].grid(True)
    axs[0, 1].legend()

    axs[1, 1].plot(time_ref, np.real(refocus_rf.G), label='Grad Ref', color='tab:red')
    axs[1, 1].set_title(f"Gradient")
    axs[1, 1].set_xlabel("Time (ms)")
    axs[1, 1].set_ylabel("Amplitude")
    axs[1, 1].grid(True)
    axs[1, 1].legend()

    # --- COLONNE 3 : Flip Angle Profiles ---
    axs[0, 2].plot(z_axis, excitation_rf.alpha, color='tab:blue', lw=2)
    axs[0, 2].set_title("Excitation Profile (Degrees)")
    axs[0, 2].set_xlabel("Position (cm)")
    axs[0, 2].set_ylabel("Flip Angle (Degrees)")
    axs[0, 2].grid(True)

    axs[1, 2].plot(z_axis, refocus_rf.alpha, color='tab:red', lw=2)
    axs[1, 2].set_title("Refocusing Profile (Degrees)")
    axs[1, 2].set_xlabel("Position (cm)")
    axs[1, 2].set_ylabel("Flip Angle (Degrees)")
    axs[1, 2].grid(True)

    # --- COLONNE 4 : Normalized Mt (au) ---
    # Excitation : Mt = sin(alpha)
    Mt_exc = np.sin(np.radians(excitation_rf.alpha))
    axs[0, 3].plot(z_axis, Mt_exc, color='tab:blue', lw=2)
    axs[0, 3].set_title("Excitation Profile - Mt (au)")
    axs[0, 3].set_xlabel("Position (cm)")
    axs[0, 3].set_ylabel("Mt (au)")
    axs[0, 3].set_ylim(0, 1.05)
    axs[0, 3].grid(True)

    # Refocalisation : Efficacité = sin^2(alpha/2)
    Mt_ref = np.sin(np.radians(refocus_rf.alpha / 2.0))**2
    axs[1, 3].plot(z_axis, Mt_ref, color='tab:red', lw=2)
    axs[1, 3].set_title("Refocusing Profile - Mt (au)")
    axs[1, 3].set_xlabel("Position (cm)")
    axs[1, 3].set_ylabel("Mt (au)")
    axs[1, 3].set_ylim(0, 1.05)
    axs[1, 3].grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f'Figure saved in {save_path}')
    return None

def main():
# --- AJOUT ARGPARSE ---
    parser = argparse.ArgumentParser(description="Test RF with variable bounds")
    parser.add_argument('--bound', type=float, default=2.5, help='Symmetric limit for z_bounds (cm)')
    args = parser.parse_args()
    
    bound_val = args.bound
    sim_bounds_z = (-bound_val, bound_val)
    
    # Définition des dossiers de sortie dynamiques basés sur le bound
    out_base = Path(f"./results/bound_{bound_val}_64_Grad")
    out_exc = out_base / "exc"
    out_ref = out_base / "ref"
    save_fig = out_base / f"result_bound_{bound_val}.png"
    # ----------------------

    log_dir = "./log"
    timestr = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs(log_dir, exist_ok=True)
    setup_logging(log_dir, f"pulseReading_bound{bound_val}_{timestr}.log")
    
    logger.info(f'Starting validation with sim_bounds_z = {sim_bounds_z}')

    # (Laisse tes chemins de fichiers et pulse_excitation/pulse_refocus ici comme avant)
    # excitation_file = Path('/mnt/c/d/pulseConversion/docs/Simulation_SEMC_3mmSlices/VE11E/excitation_normal.txt')
    # refocalisation_file = Path('/mnt/c/d/pulseConversion/docs/Simulation_SEMC_3mmSlices/VE11E/refocalisation_normal.txt')
    excitation_file = Path('/mnt/c/d/pulseConversion/docs/Simulation_SEMC_3mmSlices/VE11E/excitation_lowSAR_Grad.txt')
    refocalisation_file = Path('/mnt/c/d/pulseConversion/docs/Simulation_SEMC_3mmSlices/VE11E/refocalisation_lowSAR_Grad.txt')
    # excitation_file = Path('/mnt/c/d/pulseConversion/docs/Simulation_SEMC_3mmSlices/XA61/excitation_lowSAR.txt')
    # refocalisation_file = Path('/mnt/c/d/pulseConversion/docs/Simulation_SEMC_3mmSlices/XA61/refocalisation_lowSAR.txt')

    # pulse_excitation = rf.RfWaveform(tau=3.07e-3, angle=90.0, G=0.0, ref=1.0, phase=0.0, path=excitation_file, is_refocusing=False)
    # pulse_refocus = rf.RfWaveform(tau=2.95e-3, angle=180.0, G=0.0, ref=0.0, phase=90.0, path=refocalisation_file, is_refocusing=True)
    # pulse_excitation = rf.RfWaveform(tau=2.56e-3, angle=90.0, G=0.0, ref=1.0, phase=0.0, path=excitation_file, is_refocusing=False)
    # pulse_refocus = rf.RfWaveform(tau=3.84e-3, angle=180.0, G=0.0, ref=0.0, phase=90.0, path=refocalisation_file, is_refocusing=True)
    pulse_excitation = rf.RfWaveform(tau=2.56e-3, angle=90.0, G=0.0, ref=1.0, phase=0.0, path=excitation_file)
    pulse_refocus = rf.RfWaveform(tau=3.84e-3, angle=180.0, G=0.0, ref=0.0, phase=90.0, path=refocalisation_file)

    sim_points_z = 64

    logger.info("Processing Excitation Pulse...")
    excitation_rf = tool.get_RF(m_RF=pulse_excitation, m_Nrf= 3070, m_Dz=sim_bounds_z, m_Nz=sim_points_z)

    logger.info("Processing Refocusing Pulse...")
    refocus_rf = tool.get_RF(m_RF=pulse_refocus, m_Nrf=2950, m_Dz=sim_bounds_z, m_Nz=sim_points_z)
    
    print(f"Gradient moyen Excitation : {np.mean(np.abs(excitation_rf.G)):.3f} G/cm")
    print(f"Gradient moyen Refocalisation : {np.mean(np.abs(refocus_rf.G)):.3f} G/cm")
    if excitation_rf is not None and refocus_rf is not None:
        logger.info("Exporting analytical data...")
        export_test_data_to_tsv(excitation_rf, sim_bounds_z, output_dir=str(out_exc))
        export_test_data_to_tsv(refocus_rf, sim_bounds_z, output_dir=str(out_ref))
        
        # Astuce : on modifie un peu plot_rf_and_profile pour qu'il prenne le chemin de sauvegarde
        plot_rf_and_profile_dynamic(excitation_rf, refocus_rf, z_bounds=sim_bounds_z, nz=sim_points_z, save_path=save_fig)
    else:
        logger.error("An error occurred. Aborting plot.")


if __name__ == '__main__':
    main()