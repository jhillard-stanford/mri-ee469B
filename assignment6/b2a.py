"""
b2a - This function takes a b polynomial, and returns the minimum phase,
  minimum power a polynomial

Inputs:
  bc - beta polynomial coefficients

Outputs:
  aca - minimum phase alpha polynomial

Converted to Python
"""

import numpy as np
try:
    from .mag2mp import mag2mp
except ImportError:
    # Fallback for direct import
    from mag2mp import mag2mp


def b2a(bc):
    """
    Takes a b polynomial, and returns the minimum phase, minimum power a polynomial.
    
    Parameters:
    -----------
    bc : array_like
        Beta polynomial coefficients
    
    Returns:
    --------
    aca : ndarray
        Minimum phase alpha polynomial
    """
    bc = np.asarray(bc)
    n = len(bc)
    
    # Calculate minimum phase alpha
    bcp = np.zeros(n * 8, dtype=complex)
    bcp[:n] = bc
    bf = np.fft.fft(bcp)
    bfmax = np.max(np.abs(bf))
    
    if bfmax >= 1.0:  # PM can result in abs(beta)>1, not physical
        # Scale it so that abs(beta)<1 so that alpha will be analytic
        bf = bf / (1e-8 + bfmax)
    
    afa = mag2mp(np.sqrt(1 - bf * np.conj(bf)))
    aca = np.fft.fft(afa) / (n * 8)
    aca = aca[n-1::-1]  # Reverse first n elements
    
    return aca[:n]

