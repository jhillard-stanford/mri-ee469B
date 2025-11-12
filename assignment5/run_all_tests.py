# Run all tests for assignment5 Python scripts

import sys

print("=" * 60)
print("Running all tests for assignment5 Python scripts")
print("=" * 60)
print()

tests = [
    "test_ab2inv.py",
    "test_ab2rf.py",
    "test_mag2mp.py",
    "test_b2a.py"
]

for test_file in tests:
    print(f"\n{'=' * 60}")
    print(f"Running {test_file}")
    print('=' * 60)
    try:
        with open(test_file, 'r') as f:
            code = compile(f.read(), test_file, 'exec')
            exec(code, {'__name__': '__main__'})
    except Exception as e:
        print(f"Error running {test_file}: {e}")
        import traceback
        traceback.print_exc()
    print()

print("=" * 60)
print("All tests completed.")
print("=" * 60)


