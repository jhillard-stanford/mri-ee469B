# Test program for ab2rf.py

import numpy as np
from ab2rf import ab2rf

# Test 1: Simple case
print("Test 1: Simple case")
n = 4
ac = np.array([1.0, 0.5, 0.25, 0.1], dtype=complex)
bc = np.array([0.0, 0.1, 0.2, 0.3], dtype=complex)
rf = ab2rf(ac, bc)
print(f"ac = {ac}")
print(f"bc = {bc}")
print(f"rf shape: {rf.shape}")
print(f"rf = {rf}")
print(f"rf is complex: {np.iscomplexobj(rf)}")
print()

# Test 2: Round trip test (if we have abrm and ab2ex)
print("Test 2: Check output properties")
print(f"rf length: {len(rf)}")
print(f"rf dtype: {rf.dtype}")
print(f"rf magnitude range: [{np.abs(rf).min():.4f}, {np.abs(rf).max():.4f}]")
print()

# Test 3: Different sizes
print("Test 3: Different size")
ac3 = np.array([1.0, 0.8, 0.6], dtype=complex)
bc3 = np.array([0.0, 0.1, 0.15], dtype=complex)
rf3 = ab2rf(ac3, bc3)
print(f"Input length: {len(ac3)}")
print(f"Output length: {len(rf3)}")
print(f"Lengths match: {len(ac3) == len(rf3)}")
print()

# Test 4: Complex alpha and beta
print("Test 4: Complex alpha and beta")
ac4 = np.array([1+0.1j, 0.5+0.2j], dtype=complex)
bc4 = np.array([0.1+0.05j, 0.2+0.1j], dtype=complex)
rf4 = ab2rf(ac4, bc4)
print(f"ac = {ac4}")
print(f"bc = {bc4}")
print(f"rf = {rf4}")
print()

print("All tests completed.")


