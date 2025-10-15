#!/usr/bin/env python3
"""
Python 3.14-compatible entry point for run-itp-data-gen that bypasses Hydra.
This is a simplified wrapper for the simple_lean_data_gen.yaml configuration.
"""

import sys
import os
import logging
import time
import yaml

root_dir = f"{__file__.split('itp_interface')[0]}"
if root_dir not in sys.path:
    sys.path.append(root_dir)

# Import the actual implementation
from itp_interface.main.run_tool import run_data_generation
from itp_interface.main.config import Experiments, parse_config
from itp_interface.tools.log_utils import setup_logger


def load_yaml_config(config_path: str) -> dict:
    """Load a YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def resolve_hydra_defaults(config_dir: str, config: dict) -> dict:
    """
    Manually resolve Hydra defaults by loading referenced config files.
    This is a simplified version that handles the basic case.
    """
    resolved = {}

    if 'defaults' in config:
        for default in config['defaults']:
            if isinstance(default, dict):
                # Handle dictionary-style defaults (e.g., "benchmark: simple_benchmark_lean")
                for key, value in default.items():
                    if key.startswith('override'):
                        continue  # Skip override directives
                    subconfig_path = os.path.join(config_dir, key, f"{value}.yaml")
                    if os.path.exists(subconfig_path):
                        subconfig = load_yaml_config(subconfig_path)
                        resolved[key] = subconfig
            elif isinstance(default, str):
                # Handle string-style defaults
                subconfig_path = os.path.join(config_dir, f"{default}.yaml")
                if os.path.exists(subconfig_path):
                    subconfig = load_yaml_config(subconfig_path)
                    resolved.update(subconfig)

    # Merge main config (overrides defaults)
    for key, value in config.items():
        if key != 'defaults':
            if key in resolved and isinstance(resolved[key], dict) and isinstance(value, dict):
                resolved[key].update(value)
            else:
                resolved[key] = value

    return resolved


def create_experiment_from_dict(config: dict) -> Experiments:
    """Create an Experiments object from a dictionary configuration."""
    # This is a simplified version - you may need to adjust based on your config structure
    from omegaconf import OmegaConf

    # Convert dict to OmegaConf for compatibility with parse_config
    omega_conf = OmegaConf.create(config)

    # Use the existing parse_config function
    return parse_config(omega_conf)


def main_no_hydra():
    """
    Main entry point that works without Hydra.
    Assumes simple_lean_data_gen.yaml configuration.
    """
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(
        description='Run ITP data generation (Python 3.14 compatible - no Hydra)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--config-dir',
        type=str,
        default='src/itp_interface/main/configs',
        help='Configuration directory (default: src/itp_interface/main/configs)'
    )
    parser.add_argument(
        '--config-name',
        type=str,
        default='simple_lean_data_gen',
        help='Configuration file name without .yaml extension (default: simple_lean_data_gen)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Override output directory'
    )

    args = parser.parse_args()

    # Determine config directory and file
    if os.path.isabs(args.config_dir):
        config_dir = args.config_dir
    else:
        config_dir = os.path.join(os.getcwd(), args.config_dir)

    # Handle config name with or without .yaml extension
    config_name = args.config_name
    if not config_name.endswith('.yaml'):
        config_name = f"{config_name}.yaml"
    config_file = os.path.join(config_dir, config_name)

    if not os.path.exists(config_file):
        print(f"Error: Configuration file not found: {config_file}")
        sys.exit(1)

    print(f"Loading configuration from: {config_file}")

    # Load and resolve configuration
    config = load_yaml_config(config_file)
    resolved_config = resolve_hydra_defaults(config_dir, config)

    # Override output directory if specified
    if args.output_dir:
        if 'run_settings' not in resolved_config:
            resolved_config['run_settings'] = {}
        resolved_config['run_settings']['output_dir'] = args.output_dir

    # Create experiment object
    try:
        experiment = create_experiment_from_dict(resolved_config)
    except Exception as e:
        print(f"Error creating experiment configuration: {e}")
        print(f"Config: {resolved_config}")
        sys.exit(1)

    # Setup logging
    log_dir = ".log/data_generation/benchmark/{}/{}".format(
        experiment.benchmark.name,
        time.strftime("%Y%m%d-%H%M%S")
    )
    os.makedirs(log_dir, exist_ok=True)
    abs_path = os.path.abspath(log_dir)
    print(f"Log Dir: {abs_path}")

    log_path = os.path.join(log_dir, "eval.log")
    logger = setup_logger(__name__, log_path, logging.INFO,
                         '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info(f"Pid: {os.getpid()}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Running without Hydra (Python 3.14 compatible mode)")
    logger.info(f"Running Experiment: {experiment.to_json(indent=4)}")

    # Run the data generation
    try:
        run_data_generation(experiment, log_dir, logger=logger)
        print(f"\n✓ Data generation completed successfully!")
        print(f"  Logs: {abs_path}")
    except Exception as e:
        logger.exception("Data generation failed")
        print(f"\n✗ Data generation failed: {e}")
        sys.exit(1)


def main():
    """
    Entry point that detects Python version and chooses appropriate method.
    """
    if sys.version_info >= (3, 14):
        # Python 3.14+ - use non-Hydra version
        print("Detected Python 3.14+ - using Hydra-free mode")
        main_no_hydra()
    else:
        # Python < 3.14 - try to use Hydra
        try:
            import hydra
            # Import the original main function
            from itp_interface.main.run_tool import main as hydra_main
            hydra_main()
        except ImportError:
            # Hydra not available, fall back to non-Hydra version
            print("Hydra not available - using Hydra-free mode")
            main_no_hydra()


if __name__ == "__main__":
    main()
