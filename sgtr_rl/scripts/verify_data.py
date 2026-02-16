"""Verify data setup for SGTR-RL.

This script checks that cached generations data is accessible and reports
available datasets.

Usage:
    uv run python -m sgtr_rl.scripts.verify_data
"""

import sys
from sgtr_rl.config import get_data_path, DataPathError
from sgtr_rl.config.paths import verify_data_setup


def main():
    """Verify data setup and print status."""
    print("=" * 70)
    print("SGTR-RL Data Setup Verification")
    print("=" * 70)
    print()

    status = verify_data_setup()

    print(f"Project Root: {status['project_root']}")
    print(f"Config File Exists: {status['config_file_exists']}")
    print()

    if status["data_found"]:
        print("✅ Data Found!")
        print(f"Data Path: {status['data_path']}")
        print()

        if status["datasets"]:
            print(f"Available Datasets ({len(status['datasets'])}):")
            for dataset in status["datasets"]:
                print(f"  - {dataset}")
        else:
            print("⚠️  No datasets found in data directory")
            print("This might be okay if the directory structure is different")

        print()
        print("✅ Data setup successful!")
        return 0

    else:
        print("❌ Data Not Found")
        print()
        if status["errors"]:
            print("Errors:")
            for error in status["errors"]:
                print(error)

        print()
        print("Please set up data access. See data/README.md for instructions.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
