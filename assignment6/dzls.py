"""
dzls - design a least squares filter

h = dzls(nf, tb, d1, d2)

    nf = filter length
    tb = time-bandwidth
    d1 = pass band ripple
    d2 = stop band ripple

dzls designs a least squares filter.

written by John Pauly, Feb 26, 1992
(c) Leland Stanford Junior University
Converted to Python
"""

import numpy as np
from scipy.signal import firls

try:
    from .dinf import dinf
except ImportError:
    # Fallback for direct import
    from dinf import dinf


def dzls(nf, tb, d1, d2):
    """
    Design a least squares filter.
    
    Parameters:
    -----------
    nf : int
        Filter length
    tb : float
        Time-bandwidth product
    d1 : float
        Pass band ripple
    d2 : float
        Stop band ripple
    
    Returns:
    --------
    h : ndarray
        Filter coefficients
    """
    di = dinf(d1, d2)
    w = di / tb
    f = np.array([0, (1 - w) * (tb / 2), (1 + w) * (tb / 2), nf / 2]) / (nf / 2)
    m = np.array([1, 1, 0, 0])
    w_weights = np.array([1, d1 / d2])
    
    h = firls(nf - 1, f, m, w_weights)
    
    return h

