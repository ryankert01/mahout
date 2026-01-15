#!/usr/bin/env python3
#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Unified QDP Benchmark CLI.

This script provides a single entry point for running all QDP benchmarks
with configurable presets and output options.

Usage:
    # Quick smoke test
    python run_benchmark.py --preset quick --framework mahout

    # Standard benchmark with all frameworks
    python run_benchmark.py --preset standard --framework all

    # Publication-quality benchmark with plots
    python run_benchmark.py --preset publication --output results/ --plot

    # Run specific benchmarks
    python run_benchmark.py --benchmark latency throughput --qubits 12 16
"""

import argparse
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional


@dataclass
class BenchmarkPreset:
    """Configuration preset for benchmarks."""

    name: str
    warmup_runs: int
    measurement_runs: int
    batches: int
    description: str


# Predefined presets
PRESETS = {
    "quick": BenchmarkPreset(
        name="quick",
        warmup_runs=1,
        measurement_runs=3,
        batches=10,
        description="Fast smoke test (1 warmup, 3 runs)",
    ),
    "standard": BenchmarkPreset(
        name="standard",
        warmup_runs=3,
        measurement_runs=10,
        batches=50,
        description="Standard benchmark (3 warmup, 10 runs)",
    ),
    "publication": BenchmarkPreset(
        name="publication",
        warmup_runs=5,
        measurement_runs=30,
        batches=100,
        description="Publication quality (5 warmup, 30 runs)",
    ),
}

BENCHMARKS = ["latency", "throughput", "e2e"]
FRAMEWORKS = ["mahout", "pennylane", "qiskit"]


def get_benchmark_script(benchmark: str) -> str:
    """Get the script path for a benchmark type."""
    script_dir = Path(__file__).parent
    scripts = {
        "latency": script_dir / "benchmark_latency.py",
        "throughput": script_dir / "benchmark_throughput.py",
        "e2e": script_dir / "benchmark_e2e.py",
    }
    return str(scripts[benchmark])


def build_latency_command(
    preset: BenchmarkPreset,
    frameworks: List[str],
    qubits: List[int],
    output: Optional[Path],
    plot: bool,
) -> List[str]:
    """Build command for latency benchmark."""
    cmd = [
        sys.executable,
        get_benchmark_script("latency"),
        "--warmup",
        str(preset.warmup_runs),
        "--runs",
        str(preset.measurement_runs),
        "--batches",
        str(preset.batches),
    ]

    # Add qubits
    for q in qubits:
        cmd.extend(["--qubits", str(q)])

    # Add frameworks
    for fw in frameworks:
        cmd.extend(["--frameworks", fw])

    if output:
        cmd.extend(["--output", str(output / "latency")])

    if plot:
        cmd.append("--plot")

    return cmd


def build_throughput_command(
    preset: BenchmarkPreset,
    frameworks: List[str],
    qubits: List[int],
    output: Optional[Path],
    plot: bool,
) -> List[str]:
    """Build command for throughput benchmark."""
    cmd = [
        sys.executable,
        get_benchmark_script("throughput"),
        "--warmup",
        str(preset.warmup_runs),
        "--runs",
        str(preset.measurement_runs),
        "--batches",
        str(preset.batches),
    ]

    # Add qubits
    for q in qubits:
        cmd.extend(["--qubits", str(q)])

    # Add frameworks
    for fw in frameworks:
        cmd.extend(["--frameworks", fw])

    if output:
        cmd.extend(["--output", str(output / "throughput")])

    if plot:
        cmd.append("--plot")

    return cmd


def build_e2e_command(
    preset: BenchmarkPreset,
    frameworks: List[str],
    qubits: int,
    samples: int,
    output: Optional[Path],
    plot: bool,
) -> List[str]:
    """Build command for e2e benchmark."""
    cmd = [
        sys.executable,
        get_benchmark_script("e2e"),
        "--warmup",
        str(preset.warmup_runs),
        "--runs",
        str(preset.measurement_runs),
        "--qubits",
        str(qubits),
        "--samples",
        str(samples),
    ]

    # Map framework names to e2e format
    e2e_frameworks = []
    for fw in frameworks:
        if fw == "mahout":
            e2e_frameworks.extend(["mahout-parquet", "mahout-arrow"])
        else:
            e2e_frameworks.append(fw)

    cmd.extend(["--frameworks"] + e2e_frameworks)

    if output:
        cmd.extend(["--output", str(output / "e2e")])

    if plot:
        cmd.append("--plot")

    return cmd


def run_command(cmd: List[str], dry_run: bool = False) -> int:
    """Run a benchmark command."""
    print(f"\n{'=' * 70}")
    print(f"Running: {' '.join(cmd)}")
    print("=" * 70)

    if dry_run:
        print("[DRY RUN] Would execute command")
        return 0

    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        return 130


def main():
    parser = argparse.ArgumentParser(
        description="Unified QDP Benchmark CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick smoke test with Mahout only
  python run_benchmark.py --preset quick --framework mahout

  # Standard benchmark comparing all frameworks
  python run_benchmark.py --preset standard --framework all

  # Publication benchmark with plots
  python run_benchmark.py --preset publication --output results/ --plot

  # Run only latency benchmark at specific qubit counts
  python run_benchmark.py --benchmark latency --qubits 10 12 14 16

Presets:
  quick       - Fast smoke test (1 warmup, 3 measurement runs)
  standard    - Standard benchmark (3 warmup, 10 measurement runs)
  publication - Publication quality (5 warmup, 30 measurement runs)
""",
    )

    # Benchmark selection
    parser.add_argument(
        "--benchmark",
        "-b",
        nargs="+",
        choices=BENCHMARKS + ["all"],
        default=["all"],
        help="Benchmarks to run (default: all)",
    )

    # Framework selection
    parser.add_argument(
        "--framework",
        "-f",
        nargs="+",
        choices=FRAMEWORKS + ["all"],
        default=["mahout"],
        help="Frameworks to benchmark (default: mahout)",
    )

    # Preset selection
    parser.add_argument(
        "--preset",
        "-p",
        choices=list(PRESETS.keys()),
        default="standard",
        help="Benchmark preset (default: standard)",
    )

    # Custom run counts (override preset)
    parser.add_argument(
        "--warmup",
        type=int,
        default=None,
        help="Override preset warmup runs",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=None,
        help="Override preset measurement runs",
    )
    parser.add_argument(
        "--batches",
        type=int,
        default=None,
        help="Override preset batch count",
    )

    # Qubit configuration
    parser.add_argument(
        "--qubits",
        "-q",
        nargs="+",
        type=int,
        default=[10, 12, 14, 16],
        help="Qubit counts to benchmark (default: 10 12 14 16)",
    )

    # E2E specific options
    parser.add_argument(
        "--samples",
        type=int,
        default=200,
        help="Number of samples for E2E benchmark (default: 200)",
    )

    # Output options
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output directory for results",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate plots (requires --output)",
    )

    # Execution options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running other benchmarks if one fails",
    )

    args = parser.parse_args()

    # Expand "all" options
    benchmarks = BENCHMARKS if "all" in args.benchmark else args.benchmark
    frameworks = FRAMEWORKS if "all" in args.framework else args.framework

    # Get preset and apply overrides
    preset = PRESETS[args.preset]
    if args.warmup is not None:
        preset = BenchmarkPreset(
            name=preset.name,
            warmup_runs=args.warmup,
            measurement_runs=preset.measurement_runs,
            batches=preset.batches,
            description=preset.description,
        )
    if args.runs is not None:
        preset = BenchmarkPreset(
            name=preset.name,
            warmup_runs=preset.warmup_runs,
            measurement_runs=args.runs,
            batches=preset.batches,
            description=preset.description,
        )
    if args.batches is not None:
        preset = BenchmarkPreset(
            name=preset.name,
            warmup_runs=preset.warmup_runs,
            measurement_runs=preset.measurement_runs,
            batches=args.batches,
            description=preset.description,
        )

    # Setup output directory
    output_path = None
    if args.output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(args.output) / f"benchmark_{timestamp}"
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"Results will be saved to: {output_path}")

    if args.plot and not args.output:
        print("Warning: --plot requires --output to be specified")

    # Print configuration
    print("\n" + "=" * 70)
    print("QDP BENCHMARK SUITE")
    print("=" * 70)
    print(f"Preset:      {preset.name} - {preset.description}")
    print(f"Warmup:      {preset.warmup_runs} runs")
    print(f"Measurement: {preset.measurement_runs} runs")
    print(f"Batches:     {preset.batches}")
    print(f"Qubits:      {args.qubits}")
    print(f"Benchmarks:  {benchmarks}")
    print(f"Frameworks:  {frameworks}")
    print(f"Output:      {output_path or 'None'}")
    print("=" * 70)

    # Run benchmarks
    results = {}
    for benchmark in benchmarks:
        if benchmark == "latency":
            cmd = build_latency_command(
                preset, frameworks, args.qubits, output_path, args.plot
            )
        elif benchmark == "throughput":
            cmd = build_throughput_command(
                preset, frameworks, args.qubits, output_path, args.plot
            )
        elif benchmark == "e2e":
            # E2E uses single qubit count (use max from list)
            cmd = build_e2e_command(
                preset,
                frameworks,
                max(args.qubits),
                args.samples,
                output_path,
                args.plot,
            )
        else:
            print(f"Unknown benchmark: {benchmark}")
            continue

        returncode = run_command(cmd, args.dry_run)
        results[benchmark] = returncode

        if returncode != 0 and not args.continue_on_error:
            print(f"\nBenchmark '{benchmark}' failed with code {returncode}")
            if not args.dry_run:
                sys.exit(returncode)

    # Summary
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    for benchmark, code in results.items():
        status = "PASSED" if code == 0 else f"FAILED ({code})"
        print(f"  {benchmark:12s} {status}")

    if output_path:
        print(f"\nResults saved to: {output_path}")

    # Exit with failure if any benchmark failed
    if any(code != 0 for code in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
