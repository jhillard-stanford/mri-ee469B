import numpy as np

def abrm(rf, g=None, x=None, y=None):
    """
    [a b] = abrm(rf,[g],[x [,y])
    
    Simulate an rf pulse, returning the Cayley-Klein parameters, alpha and beta.
    This is the Python version of abr(), which is compiled and faster.
    
    Parameters:
    -----------
    rf : array_like
        RF scaled so that sum(rf) = flip angle
    g : array_like, optional
        Gradient waveform, scaled so that (gamma/2*pi)*sum(g) = k in cycles/cm
    x : array_like, optional
        Position vector
    y : array_like, optional
        Position vector for 2D pulses (assumes imag(g) = gy)
    
    Returns:
    --------
    a : ndarray
        Alpha parameter (Cayley-Klein parameter)
    b : ndarray
        Beta parameter (Cayley-Klein parameter)
    
    Written by John Pauly, Dec 22, 2000
    (c) Board of Trustees, Leland Stanford Jr. University
    Translated from octave, and modified to scale gradient by 2pi, so k = cumsum(g) is cycles/cm
    Sept 27, 2004
    Converted to Python with NumPy
    """
    
    # Handle different input argument combinations
    if g is None and x is None:
        raise ValueError("At least one of g or x must be provided")
    elif g is None:
        # If only rf and x are provided, set g to default
        g = np.ones(len(rf)) * 2 * np.pi / len(rf)
        y = 0
    elif x is None:
        # If only rf and g are provided, set x to g
        x = g
        g = np.ones(len(rf)) * 2 * np.pi / len(rf)
        y = 0
    elif y is None:
        y = 0
    
    # Convert to row vectors (1D arrays)
    rf = np.asarray(rf).flatten()
    g = np.asarray(g).flatten()
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    
    lx = len(x)
    ly = len(y)
    
    # Initialize output arrays
    a = np.zeros((lx, ly), dtype=complex)
    b = np.zeros((lx, ly), dtype=complex)
    
    # Main computation loop
    for jj in range(ly):
        for kk in range(lx):
            # Make sure om isn't exactly zero, so n doesn't blow up
            om = x[kk] * np.real(g) + y[jj] * np.imag(g)
            om = om + (np.abs(om) < np.finfo(float).eps) * np.finfo(float).eps
            
            phi = np.sqrt(rf * np.conj(rf) + om**2)
            
            # Compute n vector components
            n = np.array([
                np.real(rf) / phi,
                np.imag(rf) / phi,
                om / phi
            ])
            
            # Compute av and bv
            av = np.cos(phi/2) - 1j * n[2, :] * np.sin(phi/2)
            bv = -1j * (n[0, :] + 1j * n[1, :]) * np.sin(phi/2)
            
            # Initialize abt
            abt = np.array([1.0, 0.0], dtype=complex)
            
            # Iterate through phi values
            for m in range(len(phi)):
                # Create rotation matrix
                rot_matrix = np.array([
                    [av[m], -np.conj(bv[m])],
                    [bv[m], np.conj(av[m])]
                ])
                abt = rot_matrix @ abt
            
            a[kk, jj] = abt[0]
            b[kk, jj] = abt[1]
    
    return a, b


def abrm_vectorized(rf, g=None, x=None, y=None):
    """
    Vectorized version of abrm across spatial positions (x, y) using NumPy broadcasting.

    Parameters:
    -----------
    rf : array_like
        RF scaled so that sum(rf) = flip angle (per time sample)
    g : array_like, optional
        Gradient waveform; real(g) interacts with x, imag(g) with y.
        Scaled such that (gamma/2*pi)*sum(g) = k in cycles/cm
    x : array_like, optional
        Position vector (cm)
    y : array_like, optional
        Position vector for 2D pulses (assumes imag(g) = gy)

    Returns:
    --------
    a : ndarray
        Alpha parameter, shape (len(x), len(y)) for 2D or (len(x),) if y is scalar
    b : ndarray
        Beta parameter, same shape as a
    """

    if g is None and x is None:
        raise ValueError("At least one of g or x must be provided")
    elif g is None:
        g = np.ones(len(rf)) * 2 * np.pi / len(rf)
        y = 0
    elif x is None:
        x = g
        g = np.ones(len(rf)) * 2 * np.pi / len(rf)
        y = 0
    elif y is None:
        y = 0

    rf = np.asarray(rf).flatten()
    g = np.asarray(g).flatten()
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()

    # Ensure gradient has real and imaginary parts for x and y encoding
    gx = np.real(g).astype(float)
    gy = np.imag(g).astype(float)

    lx = len(x)
    ly = len(y)
    nt = len(rf)

    # Broadcast to (lx, ly, nt)
    # Off-resonance term per time sample and spatial location
    om = (x[:, None, None] * gx[None, None, :]) + (y[None, :, None] * gy[None, None, :])

    # Avoid divide-by-zero: add eps where om is exactly 0
    eps = np.finfo(float).eps
    om = om + (np.abs(om) < eps) * eps

    rf_real = np.real(rf)[None, None, :]
    rf_imag = np.imag(rf)[None, None, :]
    rf_abs2 = (rf * np.conj(rf)).real[None, None, :]

    phi = np.sqrt(rf_abs2 + om**2)

    # n vector components
    n0 = rf_real / phi
    n1 = rf_imag / phi
    n2 = om / phi

    half_phi = 0.5 * phi
    c = np.cos(half_phi)
    s = np.sin(half_phi)

    av = c - 1j * n2 * s
    bv = -1j * (n0 + 1j * n1) * s

    # Initialize Cayley-Klein vector over the spatial grid
    a = np.ones((lx, ly), dtype=complex)
    b = np.zeros((lx, ly), dtype=complex)

    # Sequential time-step product, vectorized over (lx, ly)
    for m in range(nt):
        avm = av[:, :, m]
        bvm = bv[:, :, m]
        a_new = avm * a - np.conj(bvm) * b
        b_new = bvm * a + np.conj(avm) * b
        a, b = a_new, b_new

    return a, b


# Example usage and test
if __name__ == "__main__":
    # Test with simple parameters
    rf = np.array([0.1, 0.2, 0.1])  # Simple RF pulse
    x = np.array([-1, 0, 1])  # Position vector
    
    print("Testing abrm function:")
    print(f"rf = {rf}")
    print(f"x = {x}")
    
    try:
        a, b = abrm(rf, x=x)
        print(f"a shape: {a.shape}")
        print(f"b shape: {b.shape}")
        print(f"a = {a}")
        print(f"b = {b}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test with gradient
    g = np.array([0.1, 0.2, 0.1])
    print(f"\nWith gradient g = {g}")
    try:
        a, b = abrm(rf, g=g, x=x)
        print(f"a = {a}")
        print(f"b = {b}")
    except Exception as e:
        print(f"Error: {e}")
