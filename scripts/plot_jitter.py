"""
plot_jitter.py - Visualize 1-Euro Filter performance comparing raw vs. filtered coordinates.
Generates a side-by-side comparison plot suitable for documentation.

Usage:
    python scripts/plot_jitter.py [--csv data.csv] [--output docs/jitter_comparison.png]
"""

import os
import sys
import argparse
import csv
import math

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def generate_synthetic_data(num_points=120):
    """Generates synthetic slow pen stroke with webcam micro-jitter and 1-Euro smoothing."""
    from hand_tracker import PointFilter2D
    import random

    random.seed(42)
    filter_2d = PointFilter2D(min_cutoff=0.8, beta=0.02)
    rows = []

    dt = 1.0 / 30.0  # 30 FPS
    for i in range(num_points):
        t = i * dt
        # Smooth base path (gentle curve)
        base_x = 200.0 + 300.0 * math.sin(t * 1.5)
        base_y = 150.0 + 100.0 * math.cos(t * 1.5)

        # Micro-tremor / webcam sensor jitter
        jitter_x = random.gauss(0, 3.2)
        jitter_y = random.gauss(0, 3.2)
        raw_x = base_x + jitter_x
        raw_y = base_y + jitter_y

        fx, fy = filter_2d.filter(raw_x, raw_y, t)
        rows.append({
            "time": round(t, 4),
            "raw_x": round(raw_x, 2),
            "raw_y": round(raw_y, 2),
            "filtered_x": round(fx, 2),
            "filtered_y": round(fy, 2),
        })

    return rows


def load_csv(csv_path):
    """Loads trajectory data from a CSV file."""
    rows = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "time": float(r.get("time", r.get("timestamp", 0))),
                "raw_x": float(r["raw_x"]),
                "raw_y": float(r["raw_y"]),
                "filtered_x": float(r["filtered_x"]),
                "filtered_y": float(r["filtered_y"]),
            })
    return rows


def plot_jitter(data, output_path):
    """Plots raw vs. filtered trajectory comparisons and saves figure."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "Error: matplotlib is required to run plot_jitter.py.\n"
            "Install it via: pip install -r requirements-dev.txt",
            file=sys.stderr,
        )
        sys.exit(1)

    times = [d["time"] for d in data]
    raw_x = [d["raw_x"] for d in data]
    filtered_x = [d["filtered_x"] for d in data]
    raw_y = [d["raw_y"] for d in data]
    filtered_y = [d["filtered_y"] for d in data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)
    fig.patch.set_facecolor("#18181c")

    # Subplot 1: Coordinate Signal Over Time
    ax1.set_facecolor("#222228")
    ax1.plot(times, raw_x, label="Raw Sensor (with jitter)", color="#ff6b6b", alpha=0.55, linewidth=1.2)
    ax1.plot(times, filtered_x, label="1-Euro Filtered", color="#4ecdc4", linewidth=2.0)
    ax1.set_title("X-Coordinate Stabilization Over Time", color="white", fontsize=12, pad=10)
    ax1.set_xlabel("Time (seconds)", color="#cccccc", fontsize=10)
    ax1.set_ylabel("Pixel Coordinate (X)", color="#cccccc", fontsize=10)
    ax1.tick_params(colors="#aaaaaa")
    ax1.legend(facecolor="#2c2c34", edgecolor="none", labelcolor="white")
    ax1.grid(True, linestyle="--", alpha=0.2, color="#888888")

    # Subplot 2: 2D Spatial Motion Trajectory
    ax2.set_facecolor("#222228")
    ax2.plot(raw_x, raw_y, label="Raw Path", color="#ff6b6b", alpha=0.5, linewidth=1.2, linestyle=":")
    ax2.plot(filtered_x, filtered_y, label="Filtered Path", color="#ffe66d", linewidth=2.2)
    ax2.set_title("2D Spatial Trajectory Comparison", color="white", fontsize=12, pad=10)
    ax2.set_xlabel("X (pixels)", color="#cccccc", fontsize=10)
    ax2.set_ylabel("Y (pixels)", color="#cccccc", fontsize=10)
    ax2.tick_params(colors="#aaaaaa")
    ax2.legend(facecolor="#2c2c34", edgecolor="none", labelcolor="white")
    ax2.grid(True, linestyle="--", alpha=0.2, color="#888888")
    ax2.invert_yaxis()  # Match screen coordinates

    plt.tight_layout()

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    plt.savefig(output_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"[plot_jitter] Successfully saved plot to: {os.path.abspath(output_path)}")


def main():
    parser = argparse.ArgumentParser(description="Plot raw vs. 1-Euro filtered jitter comparisons.")
    parser.add_argument("--csv", type=str, default=None, help="Path to input CSV file with coordinate data")
    parser.add_argument("--output", type=str, default="docs/jitter_comparison.png", help="Path to output image file")
    parser.add_argument("--generate-sample", action="store_true", help="Generate synthetic sample data if no CSV is provided")

    args = parser.parse_args()

    if args.csv and os.path.exists(args.csv):
        print(f"[plot_jitter] Loading data from {args.csv}...")
        data = load_csv(args.csv)
    else:
        print("[plot_jitter] Generating synthetic trajectory with simulated sensor jitter...")
        data = generate_synthetic_data()

    plot_jitter(data, args.output)


if __name__ == "__main__":
    main()
