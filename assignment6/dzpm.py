"""
dzpm - design a Parks-McClellan filter

h = dzpm(nf, tb, d1, d2)

    nf = filter length
    tb = time-bandwidth
    d1 = pass band ripple
    d2 = stop band ripple

dzpm designs a Parks-McClellan filter.

written by John Pauly, Feb 26, 1992
(c) Leland Stanford Junior University
Converted to Python
"""

import numpy as np
from scipy.signal import remez

try:
    from .dinf import dinf
except ImportError:
    # Fallback for direct import
    from dinf import dinf


def dzpm(nf, tb, d1, d2):
    """
    Design a Parks-McClellan filter.
    
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
    
    # firpm in MATLAB is equivalent to remez in scipy
    h = remez(nf - 1, f, m, w_weights)
    
    return h

