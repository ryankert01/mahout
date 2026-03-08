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
Sweep benchmark: how qubits and batch size affect PyTorch vs Rust+CUDA performance.

Runs encoding benchmarks across a grid of (qubits, batch_size) values and
produces graphs showing where each backend wins.

Usage:
    python benchmark/benchmark_sweep.py
    python benchmark/benchmark_sweep.py --encoding amplitude --output sweep.png
    python benchmark/benchmark_sweep.py --qubits 2 4 8 12 16 20 --batch-sizes 8 32 64 256 1024
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import numpy as np
import torch

try:
    import pennylane as qml

    HAS_PENNYLANE = True
except ImportError:
    HAS_PENNYLANE = False


def _has_rust_engine() -> bool:
    try:
        import _qdp

        return hasattr(_qdp, "QdpEngine")
    except ImportError:
        return False


def _has_cuda() -> bool:
    return torch.cuda.is_available()


def _sample_size(num_qubits: int, encoding_method: str) -> int:
    if encoding_method == "amplitude":
        return 1 << num_qubits
    elif encoding_method == "basis":
        return 1
    elif encoding_method in ("angle", "iqp-z"):
        return num_qubits
    elif encoding_method == "iqp":
        n = num_qubits
        return n + n * (n - 1) // 2
    raise ValueError(f"Unknown encoding: {encoding_method}")


def _make_batch(batch_size: int, num_qubits: int, encoding_method: str) -> np.ndarray:
    ss = _sample_size(num_qubits, encoding_method)
    if encoding_method == "basis":
        return np.random.randint(0, 1 << num_qubits, size=(batch_size, 1)).astype(
            np.float64
        )
    return np.random.randn(batch_size, ss).astype(np.float64)


# ---------------------------------------------------------------------------
# Backend runners — return vectors/sec
# ---------------------------------------------------------------------------


def bench_pytorch(
    num_qubits: int,
    batch_size: int,
    encoding_method: str,
    num_batches: int,
    warmup: int,
) -> float:
    from qumat_qdp.encodings import encode_batch

    device = torch.device("cuda" if _has_cuda() else "cpu")
    for _ in range(warmup):
        data = torch.from_numpy(_make_batch(batch_size, num_qubits, encoding_method))
        encode_batch(
            data, num_qubits, encoding_method, precision="float32", device=device
        )
    if _has_cuda():
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(num_batches):
        data = torch.from_numpy(_make_batch(batch_size, num_qubits, encoding_method))
        encode_batch(
            data, num_qubits, encoding_method, precision="float32", device=device
        )
    if _has_cuda():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return (num_batches * batch_size) / elapsed


def bench_rust(
    num_qubits: int,
    batch_size: int,
    encoding_method: str,
    num_batches: int,
    warmup: int,
) -> float | None:
    if not _has_rust_engine():
        return None
    import _qdp

    engine = _qdp.QdpEngine(device_id=0)
    for _ in range(warmup):
        data = _make_batch(batch_size, num_qubits, encoding_method)
        try:
            r = engine.encode(
                data, num_qubits=num_qubits, encoding_method=encoding_method
            )
            _ = torch.from_dlpack(r)
        except Exception:
            return None
    if _has_cuda():
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(num_batches):
        data = _make_batch(batch_size, num_qubits, encoding_method)
        r = engine.encode(data, num_qubits=num_qubits, encoding_method=encoding_method)
        _ = torch.from_dlpack(r)
    if _has_cuda():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return (num_batches * batch_size) / elapsed


def bench_pennylane(
    num_qubits: int,
    batch_size: int,
    encoding_method: str,
    num_batches: int,
    warmup: int,
) -> float | None:
    if not HAS_PENNYLANE:
        return None
    if encoding_method not in ("amplitude", "angle", "basis"):
        return None

    dev = qml.device("default.qubit", wires=num_qubits)

    if encoding_method == "amplitude":

        @qml.qnode(dev, interface="torch")
        def circuit(inputs):
            qml.AmplitudeEmbedding(
                features=inputs, wires=range(num_qubits), normalize=True, pad_with=0.0
            )
            return qml.state()

    elif encoding_method == "angle":

        @qml.qnode(dev, interface="torch")
        def circuit(inputs):
            qml.AngleEmbedding(features=inputs, wires=range(num_qubits), rotation="Y")
            return qml.state()

    elif encoding_method == "basis":

        @qml.qnode(dev, interface="torch")
        def circuit(inputs):
            qml.BasisEmbedding(features=inputs, wires=range(num_qubits))
            return qml.state()

    state_len = 1 << num_qubits

    def _run_batch(data_np):
        if encoding_method == "basis":
            for row in data_np:
                idx = int(row[0]) % state_len
                bits = [int(b) for b in format(idx, f"0{num_qubits}b")]
                _ = circuit(bits)
        else:
            batch_t = torch.tensor(data_np, dtype=torch.float64)
            try:
                _ = circuit(batch_t)
            except Exception:
                for x in batch_t:
                    _ = circuit(x)

    for _ in range(warmup):
        try:
            _run_batch(_make_batch(batch_size, num_qubits, encoding_method))
        except Exception:
            return None

    t0 = time.perf_counter()
    for _ in range(num_batches):
        _run_batch(_make_batch(batch_size, num_qubits, encoding_method))
    elapsed = time.perf_counter() - t0
    return (num_batches * batch_size) / elapsed


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_results(
    records: list[dict[str, Any]],
    encoding_method: str,
    output_path: str,
):
    import matplotlib.pyplot as plt

    qubits_list = sorted(set(r["qubits"] for r in records))
    batch_sizes = sorted(set(r["batch_size"] for r in records))
    backends = sorted(set(r["backend"] for r in records))

    backend_colors = {
        "pytorch": "#2196F3",
        "rust+cuda": "#FF9800",
        "pennylane": "#4CAF50",
    }
    backend_markers = {"pytorch": "o", "rust+cuda": "s", "pennylane": "^"}
    backend_labels = {
        "pytorch": "PyTorch-native",
        "rust+cuda": "Rust+CUDA",
        "pennylane": "PennyLane",
    }

    has_multiple_batches = len(batch_sizes) > 1

    # --- Figure 1: Throughput vs qubits (one line per backend, per batch size) ---
    # --- Figure 2: Speedup heatmap (PyTorch / Rust) ---
    # --- Figure 3: Throughput vs batch size ---
    num_plots = 1 + has_multiple_batches + ("rust+cuda" in backends)
    fig, axes = plt.subplots(1, num_plots, figsize=(7 * num_plots, 5.5))
    if num_plots == 1:
        axes = [axes]

    plot_idx = 0

    # --- Plot 1: Throughput vs Qubits (one subplot per batch_size, or pick median) ---
    ax = axes[plot_idx]
    plot_idx += 1

    # Pick a representative batch size (median of the list)
    repr_bs = batch_sizes[len(batch_sizes) // 2]

    for be in backends:
        xs, ys = [], []
        for nq in qubits_list:
            matching = [
                r
                for r in records
                if r["backend"] == be
                and r["qubits"] == nq
                and r["batch_size"] == repr_bs
            ]
            if matching:
                xs.append(nq)
                ys.append(matching[0]["vps"])
        if xs:
            ax.plot(
                xs,
                ys,
                marker=backend_markers.get(be, "o"),
                color=backend_colors.get(be, "gray"),
                label=backend_labels.get(be, be),
                linewidth=2,
                markersize=7,
            )
    ax.set_xlabel("Qubits", fontsize=12)
    ax.set_ylabel("Vectors / sec", fontsize=12)
    ax.set_title(f"Throughput vs Qubits (batch={repr_bs})", fontsize=13)
    ax.set_yscale("log")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # --- Plot 2 (optional): Throughput vs Batch Size ---
    if has_multiple_batches:
        ax2 = axes[plot_idx]
        plot_idx += 1

        repr_nq = qubits_list[len(qubits_list) // 2]
        for be in backends:
            xs, ys = [], []
            for bs in batch_sizes:
                matching = [
                    r
                    for r in records
                    if r["backend"] == be
                    and r["qubits"] == repr_nq
                    and r["batch_size"] == bs
                ]
                if matching:
                    xs.append(bs)
                    ys.append(matching[0]["vps"])
            if xs:
                ax2.plot(
                    xs,
                    ys,
                    marker=backend_markers.get(be, "o"),
                    color=backend_colors.get(be, "gray"),
                    label=backend_labels.get(be, be),
                    linewidth=2,
                    markersize=7,
                )
        ax2.set_xlabel("Batch Size", fontsize=12)
        ax2.set_ylabel("Vectors / sec", fontsize=12)
        ax2.set_title(f"Throughput vs Batch Size (qubits={repr_nq})", fontsize=13)
        ax2.set_xscale("log", base=2)
        ax2.set_yscale("log")
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

    # --- Plot 3 (optional): Speedup heatmap PyTorch/Rust ---
    if "rust+cuda" in backends:
        ax3 = axes[plot_idx]
        plot_idx += 1

        grid = np.full((len(batch_sizes), len(qubits_list)), np.nan)
        for bi, bs in enumerate(batch_sizes):
            for qi, nq in enumerate(qubits_list):
                pt = [
                    r
                    for r in records
                    if r["backend"] == "pytorch"
                    and r["qubits"] == nq
                    and r["batch_size"] == bs
                ]
                ru = [
                    r
                    for r in records
                    if r["backend"] == "rust+cuda"
                    and r["qubits"] == nq
                    and r["batch_size"] == bs
                ]
                if pt and ru and ru[0]["vps"] > 0:
                    grid[bi, qi] = pt[0]["vps"] / ru[0]["vps"]

        if not np.all(np.isnan(grid)):
            vmin = np.nanmin(grid)
            vmax = np.nanmax(grid)
            # Center colormap at 1.0 (equal performance)
            abs_max = max(abs(np.log2(vmin)), abs(np.log2(vmax)), 0.5)
            im = ax3.imshow(
                np.log2(grid),
                aspect="auto",
                cmap="RdYlGn",
                vmin=-abs_max,
                vmax=abs_max,
                origin="lower",
            )
            ax3.set_xticks(range(len(qubits_list)))
            ax3.set_xticklabels(qubits_list)
            ax3.set_yticks(range(len(batch_sizes)))
            ax3.set_yticklabels(batch_sizes)
            ax3.set_xlabel("Qubits", fontsize=12)
            ax3.set_ylabel("Batch Size", fontsize=12)
            ax3.set_title("PyTorch / Rust+CUDA speedup", fontsize=13)

            # Annotate cells
            for bi in range(len(batch_sizes)):
                for qi in range(len(qubits_list)):
                    val = grid[bi, qi]
                    if not np.isnan(val):
                        color = (
                            "white" if abs(np.log2(val)) > abs_max * 0.6 else "black"
                        )
                        ax3.text(
                            qi,
                            bi,
                            f"{val:.1f}x",
                            ha="center",
                            va="center",
                            fontsize=9,
                            fontweight="bold",
                            color=color,
                        )

            cbar = fig.colorbar(im, ax=ax3, shrink=0.8)
            cbar.set_label("log2(speedup)  [green=PyTorch wins]", fontsize=10)

    fig.suptitle(
        f"QDP Encoding Sweep — {encoding_method}",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nGraph saved to: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Sweep benchmark: qubits x batch_size grid for PyTorch vs Rust+CUDA vs PennyLane"
    )
    parser.add_argument(
        "--qubits",
        type=int,
        nargs="+",
        default=[2, 4, 6, 8, 10, 12, 14, 16],
        help="Qubit counts to sweep (default: 2 4 6 8 10 12 14 16)",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[8, 32, 64, 256, 1024],
        help="Batch sizes to sweep (default: 8 32 64 256 1024)",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="amplitude",
        help="Encoding method (default: amplitude)",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=20,
        help="Batches per measurement (default: 20)",
    )
    parser.add_argument(
        "--warmup", type=int, default=3, help="Warmup batches (default: 3)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_sweep.png",
        help="Output graph path (default: benchmark_sweep.png)",
    )
    parser.add_argument(
        "--no-pennylane",
        action="store_true",
        help="Skip PennyLane (it can be very slow at high qubits)",
    )
    parser.add_argument(
        "--pennylane-max-qubits",
        type=int,
        default=10,
        help="Max qubits for PennyLane (default: 10, it gets very slow beyond this)",
    )
    args = parser.parse_args()

    has_rust = _has_rust_engine()
    has_pl = HAS_PENNYLANE and not args.no_pennylane

    print("QDP Encoding Sweep Benchmark")
    print(f"  CUDA: {_has_cuda()}, Rust: {has_rust}, PennyLane: {has_pl}")
    print(f"  Encoding: {args.encoding}")
    print(f"  Qubits: {args.qubits}")
    print(f"  Batch sizes: {args.batch_sizes}")
    print(f"  Batches per point: {args.num_batches}, warmup: {args.warmup}")
    print()

    total = len(args.qubits) * len(args.batch_sizes)
    records: list[dict[str, Any]] = []
    done = 0

    for nq in args.qubits:
        for bs in args.batch_sizes:
            done += 1
            tag = f"[{done}/{total}] q={nq}, bs={bs}"

            # PyTorch
            vps = bench_pytorch(nq, bs, args.encoding, args.num_batches, args.warmup)
            records.append(
                {"backend": "pytorch", "qubits": nq, "batch_size": bs, "vps": vps}
            )
            pt_vps = vps

            # Rust
            rust_str = ""
            if has_rust:
                vps_r = bench_rust(nq, bs, args.encoding, args.num_batches, args.warmup)
                if vps_r is not None:
                    records.append(
                        {
                            "backend": "rust+cuda",
                            "qubits": nq,
                            "batch_size": bs,
                            "vps": vps_r,
                        }
                    )
                    ratio = pt_vps / vps_r
                    rust_str = f"  rust={vps_r:.0f} v/s  PT/Rust={ratio:.2f}x"
                else:
                    rust_str = "  rust=FAIL"

            # PennyLane
            pl_str = ""
            if has_pl and nq <= args.pennylane_max_qubits:
                vps_p = bench_pennylane(
                    nq, bs, args.encoding, args.num_batches, args.warmup
                )
                if vps_p is not None:
                    records.append(
                        {
                            "backend": "pennylane",
                            "qubits": nq,
                            "batch_size": bs,
                            "vps": vps_p,
                        }
                    )
                    pl_str = f"  pl={vps_p:.0f} v/s  PT/PL={pt_vps / vps_p:.1f}x"

            print(f"  {tag}  pytorch={pt_vps:.0f} v/s{rust_str}{pl_str}")

    # Print summary table
    print("\n" + "=" * 90)
    print("RESULTS TABLE (vectors/sec)")
    print("=" * 90)
    backends_present = sorted(set(r["backend"] for r in records))
    header = f"{'Qubits':>6} {'Batch':>6}"
    for be in backends_present:
        lbl = {
            "pytorch": "PyTorch",
            "rust+cuda": "Rust+CUDA",
            "pennylane": "PennyLane",
        }.get(be, be)
        header += f" {lbl:>14}"
    if "rust+cuda" in backends_present:
        header += f" {'PT/Rust':>10}"
    print(header)
    print("-" * 90)

    for nq in args.qubits:
        for bs in args.batch_sizes:
            line = f"{nq:>6} {bs:>6}"
            vals = {}
            for be in backends_present:
                m = [
                    r
                    for r in records
                    if r["backend"] == be
                    and r["qubits"] == nq
                    and r["batch_size"] == bs
                ]
                if m:
                    vals[be] = m[0]["vps"]
                    line += f" {m[0]['vps']:>14.0f}"
                else:
                    line += f" {'N/A':>14}"
            if (
                "rust+cuda" in backends_present
                and "pytorch" in vals
                and "rust+cuda" in vals
            ):
                ratio = vals["pytorch"] / vals["rust+cuda"]
                line += f" {ratio:>9.2f}x"
            elif "rust+cuda" in backends_present:
                line += f" {'':>10}"
            print(line)

    # Plot
    try:
        plot_results(records, args.encoding, args.output)
    except ImportError:
        print(
            "\nmatplotlib not installed — skipping graph. Install with: pip install matplotlib"
        )
    except Exception as exc:
        print(f"\nFailed to generate graph: {exc}")


if __name__ == "__main__":
    main()
