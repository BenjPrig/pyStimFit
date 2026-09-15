# pyStimFit - RF Tools Module

A Python suite adapted from the StimFit toolbox for loading, processing, and validating MRI Radiofrequency (RF) pulse waveforms via vectorized Bloch simulations.

---

## Features

- **Bloch Simulator**: Full non-linear Bloch equation solver handling both constant and dynamic (VERSE) gradient waveforms, RF phase, and refocusing gradient lobes.
- **Slice Profile Evaluation**: Compute spatial flip angle profiles ($\alpha(z)$) and transverse magnetization ($M_t$).
- **Data Export & Visualization**: Export resampled profiles to TSV files and plot waveforms, gradients, and flip angle distributions.

---

## File Overview

| File | Description |
| :--- | :--- |
| `class_rf.py` | Defines the `RfWaveform` dataclass storing waveform data, spatial profiles, and MRI physical parameters. |
| `rf_Constructor.py` | Low-level parser functions for vendor-specific RF file formats and linear interpolation routines. |
| `pulse_sim.py` | Vectorized Bloch simulation engine supporting RF nutation, gradient precession, and rephasing lobes. |
| `rf_Tools.py` | Core pipeline functions (`read_RF`, `scale_RF`, `calc_FlipRF`, `get_RF`) bridging file parsing and simulation. |
| `main_pyRF.py` | Execution & validation script: loads excitation and refocusing pulses, runs Bloch simulations, exports TSV data, and outputs diagnostic plots. |

---

## Quickstart

WIP

## TODO 

- Translate all comment in english
- Add the consideration of B1 a priori map 

