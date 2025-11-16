#!/usr/bin/env python3
"""
Analyze Collected Data - Visualization and Analysis
====================================================

Analyzes data from test_smolvla_data_collection.py:
- Plots action trajectories
- Compares commanded vs actual robot positions
- Shows camera snapshots in grid
- Verifies denormalization
- Generates analysis report

Usage:
    python analyze_collected_data.py data_collection_<timestamp>/
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from PIL import Image


def analyze_data_collection(data_dir):
    """Analyze collected data and generate report"""

    data_dir = Path(data_dir)

    if not data_dir.exists():
        print(f"❌ Directory not found: {data_dir}")
        return

    print("="*60)
    print(f"Analyzing Data Collection")
    print("="*60)
    print(f"Directory: {data_dir}")
    print()

    # Load diagnostic log
    diagnostic_file = data_dir / "diagnostic_log.json"
    if diagnostic_file.exists():
        with open(diagnostic_file, 'r') as f:
            diag = json.load(f)
        print("✅ Diagnostic log loaded")
        print(f"   Dataset stats mean: {diag['dataset_stats']['mean']}")
        print(f"   Dataset stats std: {diag['dataset_stats']['std']}")
    else:
        print("⚠️  diagnostic_log.json not found")
        diag = None

    # Load CSVs
    print("\n" + "="*60)
    print("Loading CSV Data")
    print("="*60)

    actions_file = data_dir / "actions.csv"
    if actions_file.exists():
        actions_df = pd.read_csv(actions_file)
        print(f"✅ actions.csv: {len(actions_df)} rows")
    else:
        print("❌ actions.csv not found")
        actions_df = None

    robot_state_file = data_dir / "robot_state.csv"
    if robot_state_file.exists():
        state_df = pd.read_csv(robot_state_file)
        print(f"✅ robot_state.csv: {len(state_df)} rows")
    else:
        print("❌ robot_state.csv not found")
        state_df = None

    model_outputs_file = data_dir / "model_outputs.csv"
    if model_outputs_file.exists():
        model_df = pd.read_csv(model_outputs_file)
        print(f"✅ model_outputs.csv: {len(model_df)} rows")
    else:
        print("❌ model_outputs.csv not found")
        model_df = None

    # Count snapshots
    snapshots_dir = data_dir / "snapshots"
    if snapshots_dir.exists():
        snapshots = list(snapshots_dir.glob("*.jpg"))
        camera1_snaps = len([s for s in snapshots if 'camera1' in s.name])
        camera2_snaps = len([s for s in snapshots if 'camera2' in s.name])
        print(f"✅ Snapshots: {camera1_snaps} (camera1), {camera2_snaps} (camera2)")
    else:
        print("❌ snapshots/ directory not found")
        camera1_snaps = camera2_snaps = 0

    # Create analysis plots
    print("\n" + "="*60)
    print("Generating Plots")
    print("="*60)

    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle(f"Data Collection Analysis\n{data_dir.name}", fontsize=14)

    # Plot 1: Commanded Actions Over Time
    if actions_df is not None:
        ax = axes[0, 0]
        joints = ['shoulder_pan', 'shoulder_lift', 'elbow_flex',
                 'wrist_flex', 'wrist_roll', 'gripper']
        for joint in joints:
            if joint in actions_df.columns:
                ax.plot(actions_df['timestamp'], actions_df[joint], label=joint, alpha=0.7)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Commanded Position (degrees)')
        ax.set_title('Commanded Actions Over Time')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        print("✅ Plot: Commanded Actions")

    # Plot 2: Actual Robot State Over Time
    if state_df is not None:
        ax = axes[0, 1]
        joints = ['shoulder_pan', 'shoulder_lift', 'elbow_flex',
                 'wrist_flex', 'wrist_roll', 'gripper']
        for joint in joints:
            if joint in state_df.columns:
                ax.plot(state_df['timestamp'], state_df[joint], label=joint, alpha=0.7)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Actual Position (degrees)')
        ax.set_title('Actual Robot State Over Time')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        print("✅ Plot: Actual Robot State")

    # Plot 3: Tracking Error (Commanded vs Actual)
    if actions_df is not None and state_df is not None:
        ax = axes[1, 0]
        joints = ['shoulder_pan', 'shoulder_lift', 'elbow_flex']
        for joint in joints:
            if joint in actions_df.columns and joint in state_df.columns:
                # Align by timestamp
                error = actions_df[joint] - state_df[joint]
                ax.plot(actions_df['timestamp'], error, label=joint, alpha=0.7)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Tracking Error (degrees)')
        ax.set_title('Tracking Error (Commanded - Actual)')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        print("✅ Plot: Tracking Error")

    # Plot 4: Denormalization Verification
    if model_df is not None:
        ax = axes[1, 1]
        # Plot first 3 joints (normalized vs denormalized)
        norm_cols = ['norm_0', 'norm_1', 'norm_2']
        denorm_cols = ['denorm_0', 'denorm_1', 'denorm_2']
        labels = ['shoulder_pan', 'shoulder_lift', 'elbow_flex']

        for i, (norm, denorm, label) in enumerate(zip(norm_cols, denorm_cols, labels)):
            if norm in model_df.columns and denorm in model_df.columns:
                ax.plot(model_df['timestamp'], model_df[norm],
                       linestyle='--', alpha=0.5, color=f'C{i}')
                ax.plot(model_df['timestamp'], model_df[denorm],
                       linestyle='-', alpha=0.7, color=f'C{i}', label=label)

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Value')
        ax.set_title('Denormalization Verification\n(Dashed=normalized, Solid=denormalized)')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        print("✅ Plot: Denormalization Verification")

    # Plot 5: Action Value Ranges (verify they're in degree range)
    if actions_df is not None:
        ax = axes[2, 0]
        joints = ['shoulder_pan', 'shoulder_lift', 'elbow_flex',
                 'wrist_flex', 'wrist_roll', 'gripper']
        ranges = []
        labels = []
        for joint in joints:
            if joint in actions_df.columns:
                ranges.append([actions_df[joint].min(), actions_df[joint].max()])
                labels.append(joint.replace('_', '\n'))

        if ranges:
            ax.barh(range(len(ranges)), [r[1]-r[0] for r in ranges],
                   left=[r[0] for r in ranges], alpha=0.7)
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, fontsize=8)
            ax.set_xlabel('Position (degrees)')
            ax.set_title('Action Value Ranges\n(Should be ±100° for degrees)')
            ax.axvline(x=-100, color='r', linestyle='--', alpha=0.3, label='±100°')
            ax.axvline(x=100, color='r', linestyle='--', alpha=0.3)
            ax.grid(True, alpha=0.3, axis='x')
            print("✅ Plot: Action Value Ranges")

    # Plot 6: Camera Snapshot Preview
    ax = axes[2, 1]
    ax.axis('off')
    if camera1_snaps > 0:
        # Show first snapshot from camera1
        first_snap = sorted(snapshots_dir.glob("*camera1.jpg"))[0]
        img = Image.open(first_snap)
        ax.imshow(img)
        ax.set_title(f'Camera Snapshot Preview\n{first_snap.name}')
        print(f"✅ Plot: Camera Snapshot ({first_snap.name})")
    else:
        ax.text(0.5, 0.5, 'No snapshots available',
               ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()

    # Save plot
    plot_file = data_dir / "analysis_plots.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"\n✅ Plots saved to: {plot_file}")

    # Generate text report
    print("\n" + "="*60)
    print("Analysis Report")
    print("="*60)

    report_lines = []
    report_lines.append("="*60)
    report_lines.append("Data Collection Analysis Report")
    report_lines.append("="*60)
    report_lines.append(f"Directory: {data_dir}")
    report_lines.append("")

    # Dataset stats
    if diag and 'dataset_stats' in diag:
        report_lines.append("Dataset Stats:")
        report_lines.append(f"  Mean: {diag['dataset_stats']['mean']}")
        report_lines.append(f"  Std:  {diag['dataset_stats']['std']}")
        report_lines.append("")

    # Data files
    report_lines.append("Data Files:")
    report_lines.append(f"  Actions: {len(actions_df) if actions_df is not None else 0} rows")
    report_lines.append(f"  Robot State: {len(state_df) if state_df is not None else 0} rows")
    report_lines.append(f"  Model Outputs: {len(model_df) if model_df is not None else 0} rows")
    report_lines.append(f"  Camera1 Snapshots: {camera1_snaps}")
    report_lines.append(f"  Camera2 Snapshots: {camera2_snaps}")
    report_lines.append("")

    # Action statistics
    if actions_df is not None:
        report_lines.append("Action Statistics (Commanded):")
        joints = ['shoulder_pan', 'shoulder_lift', 'elbow_flex',
                 'wrist_flex', 'wrist_roll', 'gripper']
        for joint in joints:
            if joint in actions_df.columns:
                mean_val = actions_df[joint].mean()
                std_val = actions_df[joint].std()
                min_val = actions_df[joint].min()
                max_val = actions_df[joint].max()
                report_lines.append(f"  {joint}:")
                report_lines.append(f"    Mean: {mean_val:7.2f}°  Std: {std_val:6.2f}°")
                report_lines.append(f"    Range: [{min_val:7.2f}°, {max_val:7.2f}°]")
        report_lines.append("")

    # Tracking error analysis
    if actions_df is not None and state_df is not None:
        report_lines.append("Tracking Error Analysis:")
        joints = ['shoulder_pan', 'shoulder_lift', 'elbow_flex']
        for joint in joints:
            if joint in actions_df.columns and joint in state_df.columns:
                error = (actions_df[joint] - state_df[joint]).abs()
                mean_error = error.mean()
                max_error = error.max()
                report_lines.append(f"  {joint}: Mean={mean_error:.2f}°, Max={max_error:.2f}°")
        report_lines.append("")

    # Denormalization check
    if model_df is not None:
        report_lines.append("Denormalization Check:")
        # Check if normalized and denormalized values differ significantly
        for i in range(6):
            norm_col = f'norm_{i}'
            denorm_col = f'denorm_{i}'
            if norm_col in model_df.columns and denorm_col in model_df.columns:
                norm_vals = model_df[norm_col]
                denorm_vals = model_df[denorm_col]
                ratio = (denorm_vals / (norm_vals + 1e-8)).abs().mean()
                changed = not np.allclose(norm_vals, denorm_vals, rtol=0.01)
                status = "✅ Working" if changed else "❌ NOT Working"
                report_lines.append(f"  Joint {i}: {status} (scale ratio: {ratio:.1f}x)")
        report_lines.append("")

    # Key findings
    report_lines.append("Key Findings:")

    # Check if actions are in degree range
    if actions_df is not None:
        max_action = max([actions_df[j].abs().max() for j in joints if j in actions_df.columns])
        if max_action > 10:
            report_lines.append("  ✅ Actions in degree range (max magnitude > 10°)")
        else:
            report_lines.append("  ⚠️  Actions may be too small (max magnitude < 10°)")

    # Check if denormalization is working
    if model_df is not None:
        denorm_working = False
        for i in range(6):
            if f'norm_{i}' in model_df.columns and f'denorm_{i}' in model_df.columns:
                if not np.allclose(model_df[f'norm_{i}'], model_df[f'denorm_{i}'], rtol=0.01):
                    denorm_working = True
                    break
        if denorm_working:
            report_lines.append("  ✅ Denormalization is working")
        else:
            report_lines.append("  ❌ Denormalization may not be working")

    # Check tracking error
    if actions_df is not None and state_df is not None:
        avg_error = sum([abs(actions_df[j] - state_df[j]).mean()
                        for j in ['shoulder_pan', 'shoulder_lift', 'elbow_flex']
                        if j in actions_df.columns and j in state_df.columns]) / 3
        if avg_error < 5:
            report_lines.append(f"  ✅ Good tracking (avg error: {avg_error:.2f}°)")
        elif avg_error < 15:
            report_lines.append(f"  ⚠️  Moderate tracking error ({avg_error:.2f}°)")
        else:
            report_lines.append(f"  ❌ High tracking error ({avg_error:.2f}°)")

    report_lines.append("")
    report_lines.append("="*60)
    report_lines.append("Next Steps:")
    report_lines.append("1. Review snapshots/ to see what robot saw")
    report_lines.append("2. Check analysis_plots.png for visual analysis")
    report_lines.append("3. If actions look reasonable but task fails:")
    report_lines.append("   → Camera domain mismatch (different viewpoint)")
    report_lines.append("   → Need fine-tuning with your own data")
    report_lines.append("4. If actions look random/chaotic:")
    report_lines.append("   → Model not understanding the task")
    report_lines.append("   → Check camera images quality")
    report_lines.append("="*60)

    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    # Save report
    report_file = data_dir / "analysis_report.txt"
    with open(report_file, 'w') as f:
        f.write(report_text)
    print(f"\n✅ Report saved to: {report_file}")

    # Show plot
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_collected_data.py <data_collection_directory>")
        print("\nExample:")
        print("  python analyze_collected_data.py data_collection_20251115_143022/")
        sys.exit(1)

    analyze_data_collection(sys.argv[1])
