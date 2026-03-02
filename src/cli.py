import argparse
from typing import Optional


def parse_args(argv: Optional[list[str]] = None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device", type=str, default=None, help="Warp device, e.g. 'cuda:0' or 'cpu'"
    )

    parser.add_argument(
        "--IK", action="store_true", help="Enable inverse kinematics (Part III)"
    )
    args = parser.parse_args(argv)
    return args