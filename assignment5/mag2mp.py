"""
mag2mp - take the magnitude of the fft of a signal, and return the
  fft of the analytic signal.

If you have a waveform B that you want to the minimum phase
version of, then
   1) pad B out by adding zeros up to a factor of 4 or more
   2) compute C=abs(fft(B))
   3) compute D=mag2mp(C)
   4) compute E=ifft(D)
E is the minimum phase function, also zero padded.  The part you want
should be at the beginning.  You will see echos later on if you didn't
add enough padding.

Written by John Pauly, Oct 1989
(c) Board of Trustees, Leland Stanford Junior University
Converted to Python
"""

import numpy as np


def mag2mp(x):
    """
    Take the magnitude of the fft of a signal, and return the fft of the analytic signal.
    
    Parameters:
    -----------
    x : array_like
        Magnitude of analytic signal fft.
    
    Returns:
    --------
    a : ndarray
        FFT of analytic signal
    """
    n = len(x)
    xl = np.log(x)  # log of mag spectrum
    xlf = np.fft.fft(xl)  # FFT of log
    
    xlfp = np.zeros_like(xlf, dtype=complex)
    xlfp[0] = xlf[0]  # keep DC the same
    xlfp[1:(n//2)] = 2 * xlf[1:(n//2)]  # double positive freqs
    xlfp[n//2] = xlf[n//2]  # keep half Nyquist the same, too
    xlfp[(n//2 + 1):] = 0  # zero neg freqs
    
    xlaf = np.fft.ifft(xlfp)  # IFFT
    a = np.exp(xlaf)  # complex exponentiation
    
    return a



