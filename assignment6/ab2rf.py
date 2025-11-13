"""
ab2rf - take two polynomials for alpha and beta, and return an
  rf waveform that would generate them.

COMPLEX RF VERSION

rf = ab2rf(a, b)
  a, b - polynomials for alpha and beta
  rf - rf waveform that produces alpha and beta under the hard pulse
    approximation.

Converted to Python
"""

import numpy as np


def ab2rf(ac, bc):
    """
    Take two polynomials for alpha and beta, and return an RF waveform
    that would generate them.
    
    Parameters:
    -----------
    ac : array_like
        Alpha polynomial coefficients
    bc : array_like
        Beta polynomial coefficients
    
    Returns:
    --------
    rf : ndarray
        RF waveform that produces alpha and beta under the hard pulse
        approximation.
    """
    ac = np.asarray(ac, dtype=complex)
    bc = np.asarray(bc, dtype=complex)
    n = len(ac)
    
    rf = np.zeros(n, dtype=complex)
    j = 1j
    
    # Iterate backwards from n to 1
    for i in range(n-1, -1, -1):
        # Calculate c and s
        ratio = bc[i] / ac[i]
        c = np.sqrt(1 / (1 + np.abs(ratio)**2))
        s = np.conj(c * ratio)
        
        # Calculate theta and psi
        theta = np.arctan2(np.abs(s), c)
        psi = np.angle(s)
        
        # Calculate RF pulse
        rf[i] = 2 * (theta * np.cos(psi) + j * theta * np.sin(psi))
        
        # Update polynomials for next iteration
        if i > 0:
            acn = c * ac + s * bc
            bcn = -np.conj(s) * ac + c * bc
            # MATLAB: ac = acn(2:i) -> Python: ac = acn[1:i+1]
            # MATLAB: bc = bcn(1:i-1) -> Python: bc = bcn[0:i]
            ac = acn[1:(i+1)]
            bc = bcn[0:i]
    
    return rf

