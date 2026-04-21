"""
Test Runner
Assignment 8 - Part B: Testing

Runs all white box and black box tests with coverage reporting.
"""

import sys
import os
import subprocess
from datetime import datetime

def print_header(title):
    """Print formatted header."""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80 + "\n")


def run_white_box_tests():
    """Run white box tests with coverage."""
    print_header("WHITE BOX TESTING")
    
    print("Running white box tests with coverage...")
    print("-" * 80)
    
    # Run pytest with coverage
    cmd = [
        sys.executable, "-m", "pytest",
        "white_box/test_white_box.py",
        "-v",
        "--tb=short",
        "--cov=../Part_A_DAL/dal",
        "--cov-report=term-missing",
        "--cov-report=html:white_box/coverage_html"
    ]
    
    result = subprocess.run(cmd, cwd=os.path.dirname(__file__))
    
    print("\n" + "-" * 80)
    print(f"White box tests completed with exit code: {result.returncode}")
    
    if result.returncode == 0:
        print("Coverage report generated in: white_box/coverage_html/index.html")
    
    return result.returncode


def run_black_box_tests():
    """Run black box tests."""
    print_header("BLACK BOX TESTING")
    
    print("Running black box tests...")
    print("-" * 80)
    
    # Run pytest
    cmd = [
        sys.executable, "-m", "pytest",
        "black_box/test_black_box.py",
        "-v",
        "--tb=short"
    ]
    
    result = subprocess.run(cmd, cwd=os.path.dirname(__file__))
    
    print("\n" + "-" * 80)
    print(f"Black box tests completed with exit code: {result.returncode}")
    
    return result.returncode


def generate_summary(white_box_result, black_box_result):
    """Generate test summary."""
    print_header("TEST SUMMARY")
    
    print(f"Test Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    print("White Box Tests:")
    if white_box_result == 0:
        print("  Status: PASSED")
    else:
        print(f"  Status: FAILED (Exit code: {white_box_result})")
    
    print()
    print("Black Box Tests:")
    if black_box_result == 0:
        print("  Status: PASSED")
    else:
        print(f"  Status: FAILED (Exit code: {black_box_result})")
    
    print()
    print("-" * 80)
    
    if white_box_result == 0 and black_box_result == 0:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED - Please review the output above")
    
    print("=" * 80 + "\n")


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print(" VeriSupport Test Suite")
    print(" Assignment 8 - Part B: Testing")
    print("=" * 80)
    
    # Check if pytest is installed
    try:
        import pytest
    except ImportError:
        print("\nError: pytest is not installed")
        print("Please install it with: pip install pytest pytest-cov")
        return 1
    
    # Run tests
    white_box_result = run_white_box_tests()
    black_box_result = run_black_box_tests()
    
    # Generate summary
    generate_summary(white_box_result, black_box_result)
    
    # Return overall result
    return 0 if (white_box_result == 0 and black_box_result == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
