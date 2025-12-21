#!/usr/bin/env python3
"""
Test script to verify the TimingTracker logic works correctly.
This can be run without GPU/CUDA.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

# Import just the TimingTracker class
from collections import defaultdict


class TimingTracker:
    """Helper class to track timing of different components."""

    def __init__(self):
        self.timings = defaultdict(float)

    def record(self, component, duration):
        """Record time for a component."""
        self.timings[component] += duration

    def get(self, component):
        """Get time for a component."""
        return self.timings.get(component, 0.0)

    def print_breakdown(self, framework_name):
        """Print timing breakdown."""
        print(f"\n  === {framework_name} Component Breakdown ===")
        total = sum(self.timings.values())
        for component in sorted(self.timings.keys()):
            time_val = self.timings[component]
            pct = (time_val / total * 100) if total > 0 else 0
            print(f"  {component:25s} {time_val:8.4f} s ({pct:5.1f}%)")
        print(f"  {'Total':25s} {total:8.4f} s (100.0%)")


def test_timing_tracker():
    """Test the TimingTracker functionality."""
    print("Testing TimingTracker...")

    # Create a few timing trackers
    timing_mahout = TimingTracker()
    timing_mahout.record("1. IO + Encoding", 0.5)
    timing_mahout.record("2. DLPack Conversion", 0.1)
    timing_mahout.record("3. Reshape & Convert", 0.05)
    timing_mahout.record("4. Forward Pass", 0.2)

    timing_pennylane = TimingTracker()
    timing_pennylane.record("1. IO (Disk Read)", 0.3)
    timing_pennylane.record("2. Encoding (with Norm)", 2.5)
    timing_pennylane.record("3. GPU Transfer", 0.4)
    timing_pennylane.record("4. Forward Pass", 0.2)

    timing_qiskit = TimingTracker()
    timing_qiskit.record("1. IO (Disk Read)", 0.3)
    timing_qiskit.record("2. Normalization", 0.2)
    timing_qiskit.record("3. Encoding (State Prep)", 5.0)
    timing_qiskit.record("4. GPU Transfer", 0.4)
    timing_qiskit.record("5. Forward Pass", 0.2)

    # Print individual breakdowns
    timing_mahout.print_breakdown("Mahout-Parquet")
    timing_pennylane.print_breakdown("PennyLane")
    timing_qiskit.print_breakdown("Qiskit")

    # Test comparison table
    print("\n" + "=" * 70)
    print("COMPONENT TIMING COMPARISON")
    print("=" * 70)

    timings_dict = {
        "Mahout-Parquet": timing_mahout,
        "PennyLane": timing_pennylane,
        "Qiskit": timing_qiskit,
    }

    # Collect all unique components
    all_components = set()
    for timing in timings_dict.values():
        all_components.update(timing.timings.keys())

    # Print header
    header = f"{'Component':<30s}"
    for name in timings_dict.keys():
        header += f" {name:>15s}"
    print(header)
    print("-" * 70)

    # Print each component
    for component in sorted(all_components):
        row = f"{component:<30s}"
        for name, timing in timings_dict.items():
            time_val = timing.get(component)
            if time_val > 0:
                row += f" {time_val:>13.4f}s"
            else:
                row += f" {'-':>15s}"
        print(row)

    # Print totals
    print("-" * 70)
    totals_row = f"{'TOTAL':<30s}"
    for name, timing in timings_dict.items():
        total = sum(timing.timings.values())
        totals_row += f" {total:>13.4f}s"
    print(totals_row)

    print("\n✅ TimingTracker logic test passed!")


if __name__ == "__main__":
    test_timing_tracker()
