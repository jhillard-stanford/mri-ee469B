# Test program for b2a.py

import numpy as np
from b2a import b2a

# Test 1: Simple beta polynomial
print("Test 1: Simple beta polynomial")
bc = np.array([0.1, 0.2, 0.15, 0.05], dtype=complex)
aca = b2a(bc)
print(f"Input bc: {bc}")
print(f"Output aca shape: {aca.shape}")
print(f"Output aca: {aca}")
print(f"Output is complex: {np.iscomplexobj(aca)}")
print()

# Test 2: Real beta
print("Test 2: Real beta")
bc2 = np.array([0.2, 0.3, 0.2, 0.1])
aca2 = b2a(bc2)
print(f"Input bc: {bc2}")
print(f"Output aca: {aca2}")
print()

# Test 3: Complex beta
print("Test 3: Complex beta")
bc3 = np.array([0.1+0.05j, 0.2+0.1j, 0.15+0.08j], dtype=complex)
aca3 = b2a(bc3)
print(f"Input bc: {bc3}")
print(f"Output aca: {aca3}")
print()

# Test 4: Check output length
print("Test 4: Output length")
print(f"Input length: {len(bc)}")
print(f"Output length: {len(aca)}")
print(f"Lengths match: {len(bc) == len(aca)}")
print()

# Test 5: Small beta (should be stable)
print("Test 5: Small beta values")
bc5 = np.array([0.01, 0.02, 0.01])
aca5 = b2a(bc5)
print(f"Input bc: {bc5}")
print(f"Output aca: {aca5}")
print(f"Output is finite: {np.all(np.isfinite(aca5))}")
print()

# Test 6: Larger beta (may need scaling)
print("Test 6: Larger beta values")
bc6 = np.array([0.5, 0.6, 0.4, 0.3])
aca6 = b2a(bc6)
print(f"Input bc: {bc6}")
print(f"Output aca: {aca6}")
print(f"Output is finite: {np.all(np.isfinite(aca6))}")
print()

print("All tests completed.")


