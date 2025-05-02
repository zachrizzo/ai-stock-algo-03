#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compatibility Wrapper Module
===========================
This module provides backward compatibility for the legacy shell script wrappers.
It forwards commands to the new unified trade.py CLI.
"""

import os
import sys
import subprocess

def create_wrapper_script(strategy_name: str, shell_script_path: str) -> None:
    """
    Create a shell script wrapper that forwards to the new CLI
    
    Args:
        strategy_name: Name of the strategy (dmt_v2, tri_shot, etc.)
        shell_script_path: Path to write the shell script
    """
    script_content = f"""#!/bin/bash
# Compatibility wrapper for {strategy_name}
# This forwards to the new unified CLI at scripts/trade.py

# Get the directory of this script
DIR="$( cd "$( dirname "${{BASH_SOURCE[0]}}" )" && pwd )"

# Find the project root (parent of the directory containing this script)
PROJECT_ROOT="$( cd "$DIR/.." && pwd )"

# Forward to the new CLI
"$PROJECT_ROOT/scripts/trade.py" {strategy_name} "$@"
"""
    
    # Write the script
    with open(shell_script_path, 'w') as f:
        f.write(script_content)
    
    # Make it executable
    os.chmod(shell_script_path, 0o755)
    
    print(f"Created wrapper script at {shell_script_path}")

def main():
    """Create wrapper scripts for backward compatibility"""
    # Define the mappings between legacy names and new strategy names
    wrapper_configs = [
        {'legacy_name': 'dmt', 'new_name': 'dmt_v2'},
        {'legacy_name': 'tri_shot', 'new_name': 'tri_shot'},
        {'legacy_name': 'turbo_qt', 'new_name': 'turbo_qt'},
        {'legacy_name': 'hybrid', 'new_name': 'compare'}
    ]
    
    # Get the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Create wrapper scripts
    for config in wrapper_configs:
        legacy_path = os.path.join(project_root, config['legacy_name'])
        create_wrapper_script(config['new_name'], legacy_path)
    
    print("All wrapper scripts created successfully.")

if __name__ == '__main__':
    main()
