#!/usr/bin/env python3
# -*- coding: utf-8 -*- #
# ============================================= #
# py rfDataclass
# B. Prigent
# 18-May-2026
# ============================================= #
'''
    An RF class to have an easier way to process RF data.
'''

from dataclasses import dataclass, field
from typing import Union
import numpy as np
from pathlib import Path

@dataclass
class RfWaveform:
    '''Dataclass storing Radiofrequency (RF) pulse waveforms and associated MRI parameters.'''
    path: Path = field(default_factory=Path)
    rf_waveform: np.ndarray = field(default_factory=lambda: np.array([], dtype=complex))
    phase: float = 0.0          # Phase in degrees
    tau: float = 0.0            # Pulse duration in seconds
    G: Union[float, np.ndarray] = 0.0              # Slice selection gradient amplitude (G/cm)
    ref: float = 0.0            # Refocusing fraction
    angle: float = 0.0          # Nominal flip angle (degrees)
    alpha: np.ndarray = field(default_factory=lambda: np.array([], dtype=float)) # Real