# Test program for mag2mp.py

import numpy as np
from mag2mp import mag2mp

# Test 1: Simple magnitude spectrum
print("Test 1: Simple magnitude spectrum")
n = 64
x = np.ones(n)  # Uniform magnitude
a = mag2mp(x)
print(f"Input: ones({n})")
print(f"Output shape: {a.shape}")
print(f"Output is complex: {np.iscomplexobj(a)}")
print(f"Output magnitude range: [{np.abs(a).min():.4f}, {np.abs(a).max():.4f}]")
print()

# Test 2: Decaying magnitude
print("Test 2: Decaying magnitude")
n2 = 32
x2 = np.exp(-np.linspace(0, 2, n2))
a2 = mag2mp(x2)
print(f"Input range: [{x2.min():.4f}, {x2.max():.4f}]")
print(f"Output magnitude range: [{np.abs(a2).min():.4f}, {np.abs(a2).max():.4f}]")
print()

# Test 3: Sinc-like magnitude
print("Test 3: Sinc-like magnitude")
n3 = 128
freq = np.linspace(-np.pi, np.pi, n3)
x3 = np.abs(np.sinc(freq / np.pi))
x3 = np.maximum(x3, 1e-10)  # Avoid log(0)
a3 = mag2mp(x3)
print(f"Input range: [{x3.min():.4f}, {x3.max():.4f}]")
print(f"Output shape: {a3.shape}")
print()

# Test 4: Check output properties
print("Test 4: Output properties")
print(f"Output is complex: {np.iscomplexobj(a3)}")
print(f"Output has expected length: {len(a3) == n3}")
print(f"Output magnitude is positive: {np.all(np.abs(a3) > 0)}")
print()

# Test 5: Edge case - small values
print("Test 5: Small values")
x5 = np.array([1.0, 0.1, 0.01, 0.001]) + 1e-10
a5 = mag2mp(x5)
print(f"Input: {x5}")
print(f"Output: {a5}")
print(f"Output is finite: {np.all(np.isfinite(a5))}")
print()

print("All tests completed.")


