"""
dinf - calculate D infinity for a linear phase filter

written by John Pauly, 1992
(c) Board of Trustees, Leland Stanford Junior University
Converted to Python
"""

import numpy as np


def dinf(d1, d2):
    """
    Calculate D infinity for a linear phase filter.
    
    Parameters:
    -----------
    d1 : array_like
        Passband ripple parameter
    d2 : array_like
        Stopband ripple parameter
    
    Returns:
    --------
    d : ndarray
        D infinity value
    """
    a1 = 5.309e-3
    a2 = 7.114e-2
    a3 = -4.761e-1
    a4 = -2.66e-3
    a5 = -5.941e-1
    a6 = -4.278e-1
    
    d1 = np.asarray(d1)
    d2 = np.asarray(d2)
    
    # Handle scalar inputs
    d1_was_scalar = d1.ndim == 0
    d2_was_scalar = d2.ndim == 0
    
    # Convert to 1D arrays
    d1 = np.atleast_1d(d1)
    d2 = np.atleast_1d(d2)
    
    l10d1 = np.log10(d1)
    l10d2 = np.log10(d2)
    
    # MATLAB ensures column vectors, then does outer product
    # l10d1 as column, l10d2 as row (transposed)
    l10d1 = l10d1.reshape(-1, 1)  # Column vector
    l10d2 = l10d2.reshape(1, -1)  # Row vector
    
    l = len(d2)
    
    # MATLAB: d=(a1*l10d1.*l10d1+a2*l10d1+a3)*l10d2'+(a4*l10d1.*l10d1+a5*l10d1+a6)*ones(1,l)
    # This creates an outer product: (col vector) * (row vector) + (col vector) * (ones row)
    term1 = (a1 * l10d1**2 + a2 * l10d1 + a3)
    term2 = (a4 * l10d1**2 + a5 * l10d1 + a6)
    
    # Outer product
    d = term1 @ l10d2 + term2 @ np.ones((1, l))
    
    # If both inputs were scalars, return scalar
    if d1_was_scalar and d2_was_scalar:
        return d.item()
    
    return d

