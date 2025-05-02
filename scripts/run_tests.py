#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Runner for Stock Trader Algorithm Project
=============================================
This script runs the test suite for the refactored codebase,
ensuring all imports are working correctly.

Usage:
    python3 run_tests.py [--verbose] [--test-path TEST_PATH]
"""

import os
import sys
import argparse
import unittest
import importlib
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test-runner")

def get_test_modules():
    """Discover and return all test modules in the tests directory"""
    tests_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tests')
    test_files = []
    
    for root, _, files in os.walk(tests_dir):
        for file in files:
            if file.startswith('test_') and file.endswith('.py'):
                # Convert path to module name
                rel_path = os.path.relpath(os.path.join(root, file), 
                                          os.path.dirname(tests_dir))
                module_path = os.path.splitext(rel_path)[0].replace(os.sep, '.')
                test_files.append(module_path)
    
    return test_files

def run_import_tests():
    """Test that all core modules can be imported properly"""
    core_modules = [
        'stock_trader_o3_algo.strategies.dmt_v2_strategy',
        'stock_trader_o3_algo.strategies.market_regime',
        'stock_trader_o3_algo.data_utils.market_simulator',
        'stock_trader_o3_algo.backtester.core',
        'stock_trader_o3_algo.backtester.performance',
        'stock_trader_o3_algo.backtester.visualization'
    ]
    
    success = True
    logger.info("Testing core module imports...")
    
    for module_name in core_modules:
        try:
            module = importlib.import_module(module_name)
            logger.info(f"✓ {module_name} imported successfully")
        except ImportError as e:
            logger.error(f"✗ Failed to import {module_name}: {e}")
            success = False
    
    return success

def main():
    """Main test runner function"""
    parser = argparse.ArgumentParser(
        description="Test runner for Stock Trader Algorithm Project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--verbose', '-v', action='store_true', 
                      help='Verbose output')
    parser.add_argument('--test-path', help='Specific test path to run')
    
    args = parser.parse_args()
    
    # Add project root to Python path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)
    
    # First, run import tests
    if not run_import_tests():
        logger.error("Import tests failed. Please fix the import errors before proceeding.")
        sys.exit(1)
    
    # Discover and run tests
    if args.test_path:
        # Run specific test
        test_suite = unittest.defaultTestLoader.discover(
            start_dir=os.path.join(project_root, args.test_path),
            pattern='test_*.py'
        )
    else:
        # Run all tests
        test_suite = unittest.defaultTestLoader.discover(
            start_dir=os.path.join(project_root, 'tests'),
            pattern='test_*.py'
        )
    
    # Run tests
    verbosity = 2 if args.verbose else 1
    test_runner = unittest.TextTestRunner(verbosity=verbosity)
    result = test_runner.run(test_suite)
    
    # Return appropriate exit code
    if result.wasSuccessful():
        logger.info("All tests passed!")
        return 0
    else:
        logger.error("Tests failed!")
        return 1

if __name__ == '__main__':
    sys.exit(main())
