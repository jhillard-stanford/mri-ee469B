# Test program for ab2inv.py

import numpy as np
from ab2inv import ab2inv

# Test 1: Separate a and b arrays
print("Test 1: Separate a and b arrays")
a = np.array([1+0.5j, 2+1j, 3+1.5j], dtype=complex)
b = np.array([0.1+0.1j, 0.2+0.2j, 0.3+0.3j], dtype=complex)
mz = ab2inv(a, b)
print(f"a = {a}")
print(f"b = {b}")
print(f"mz = {mz}")
print(f"Expected: 1 - 2*|b|^2")
expected = 1 - 2 * np.abs(b)**2
print(f"Expected values: {expected}")
print(f"Match: {np.allclose(mz, expected)}")
print()

# Test 2: Concatenated array (2D)
print("Test 2: Concatenated array")
ab_combined = np.column_stack([a, b])
mz2 = ab2inv(ab_combined)
print(f"ab_combined shape: {ab_combined.shape}")
print(f"mz2 = {mz2}")
print(f"Match with Test 1: {np.allclose(mz, mz2)}")
print()

# Test 3: Real values
print("Test 3: Real values")
a_real = np.array([1.0, 2.0, 3.0])
b_real = np.array([0.1, 0.2, 0.3])
mz3 = ab2inv(a_real, b_real)
print(f"a = {a_real}")
print(f"b = {b_real}")
print(f"mz = {mz3}")
expected3 = 1 - 2 * b_real**2
print(f"Expected: {expected3}")
print(f"Match: {np.allclose(mz3, expected3)}")
print()

# Test 4: Edge case - b = 0
print("Test 4: Edge case - b = 0")
a4 = np.array([1.0, 1.0, 1.0])
b4 = np.array([0.0, 0.0, 0.0])
mz4 = ab2inv(a4, b4)
print(f"mz = {mz4}")
print(f"Expected: [1, 1, 1]")
print(f"Match: {np.allclose(mz4, [1, 1, 1])}")
print()

print("All tests completed.")


