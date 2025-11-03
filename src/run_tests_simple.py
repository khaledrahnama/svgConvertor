#!/usr/bin/env python3
"""
Simple test runner for the SVG Converter project
"""

# something
import sys
import os
import unittest


def run_tests():
    """Run all tests in the tests directory"""

    # Add the current directory to Python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)

    print("🔍 Discovering tests...")

    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = os.path.join(current_dir, 'tests')
    suite = loader.discover(start_dir, pattern='test_*.py')

    print(f"📁 Test directory: {start_dir}")
    print("🚀 Running tests...")
    print("=" * 60)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("=" * 60)

    # Print summary
    if result.wasSuccessful():
        print(f"✅ SUCCESS: All {result.testsRun} tests passed!")
        return 0
    else:
        print(f"❌ FAILURE: {len(result.failures)} failures, {len(result.errors)} errors out of {result.testsRun} tests")
        return 1


if __name__ == '__main__':
    exit_code = run_tests()
    sys.exit(exit_code)