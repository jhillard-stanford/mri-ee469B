"""
fmp - Generate an equal ripple minimum phase filter starting with a linear
  phase filter.

Called by dzmp.

written by John Pauly, 1992
(c) Board of Trustees, Leland Stanford Junior University
Converted to Python
"""

import numpy as np

try:
    from .mag2mp import mag2mp
except ImportError:
    # Fallback for direct import
    from mag2mp import mag2mp


def fmp(h):
    """
    Generate an equal ripple minimum phase filter starting with a linear
    phase filter.
    
    Parameters:
    -----------
    h : array_like
        Linear phase filter coefficients
    
    Returns:
    --------
    hmp : ndarray
        Minimum phase filter coefficients
    """
    h = np.asarray(h)
    l = len(h)
    
    if l % 2 == 0:
        raise ValueError('filter length must be odd')
    
    # Calculate next power of 2 that is >= 8*l
    lp = int(8 * 2 ** np.ceil(np.log2(l)))
    
    # Pad h to length lp
    pad_before = (lp - l) // 2
    pad_after = lp - l - pad_before
    hp = np.pad(h, (pad_before, pad_after), mode='constant', constant_values=0)
    
    # fftc is just fft in this context
    hpf = np.fft.fft(hp)
    
    # Shift minimum to be slightly above zero
    hpfs = hpf - np.min(np.real(hpf)) * 1.000001
    
    # Get minimum phase version
    hpfmp = mag2mp(np.sqrt(np.abs(hpfs)))
    
    # Convert back to time domain
    hpmp = np.fft.ifft(np.fft.fftshift(np.conj(hpfmp)))
    
    # Extract first half (minimum phase part)
    hmp = hpmp[0:(l + 1) // 2]
    
    return hmp

