"""
ab2inv - Computes the inversion profile Mz = 1 - 2*|b|^2

mz = ab2inv(a, b)  or  mz = ab2inv(ab)
  where ab is a concatenated array [a, b]

written by John Pauly, 1992
(c) Board of Trustees, Leland Stanford Junior University
Converted to Python
"""

import numpy as np


def ab2inv(a, b=None):
    """
    Computes the inversion profile Mz = 1 - 2*|b|^2
    
    Parameters:
    -----------
    a : array_like
        Alpha polynomial coefficients
    b : array_like, optional
        Beta polynomial coefficients. If None, a is assumed to be
        a concatenated array [a, b] with shape (..., 2*n)
    
    Returns:
    --------
    mz : ndarray
        Inversion profile Mz = 1 - 2*|b|^2
    """
    if b is None:
        # a is actually a concatenated [a, b] array
        a = np.asarray(a)
        n = a.shape[-1]  # last dimension
        b = a[..., (n//2):]  # second half
        a = a[..., :(n//2)]  # first half
    
    mz = 1 - 2 * np.conj(b) * b
    
    return mz


