"""
dzmp - design a minimum phase filter by first designing a linear phase
        filter and factoring it into two terms

function h = dzmp(n, tb, d1, d2)

inputs
  n  - filter length
  tb - time-bandwidth product
  d1 - passband ripple
  d2 - stopband ripple

outputs
  h - minimum phase equal-ripple filter

requires
  fmp, mag2mp

written by John Pauly, 1992
(c) Board of Trustees, Leland Stanford Junior University
Converted to Python
"""

import numpy as np
from scipy.signal import remez

try:
    from .dinf import dinf
    from .fmp import fmp
except ImportError:
    # Fallback for direct import
    from dinf import dinf
    from fmp import fmp


def dzmp(n, tb, d1, d2):
    """
    Design a minimum phase filter by first designing a linear phase
    filter and factoring it into two terms.
    
    Parameters:
    -----------
    n : int
        Filter length
    tb : float
        Time-bandwidth product
    d1 : float
        Passband ripple
    d2 : float
        Stopband ripple
    
    Returns:
    --------
    h : ndarray
        Minimum phase equal-ripple filter
    """
    n2 = 2 * n - 1
    di = 0.5 * dinf(2 * d1, 0.5 * d2 * d2)
    w = di / tb
    f = np.array([0, (1 - w) * (tb / 2), (1 + w) * (tb / 2), n / 2]) / (n / 2)
    m = np.array([1, 1, 0, 0])
    w_weights = np.array([1, 2 * d1 / (0.5 * d2 * d2)])
    
    # firpm in MATLAB is equivalent to remez in scipy
    hl = remez(n2 - 1, f, m, w_weights)
    
    h = fmp(hl)
    
    return h

