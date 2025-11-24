#!/usr/bin/env python3
"""
Fix stats.json to have per-dimension count arrays instead of single count value.
This is needed for GR00T compatibility.
"""

import json
from pathlib import Path

# Load current stats
stats_path = Path("/home/jrobot/project/XLeRobot/jdocs/top_level/datasets/meta/stats.json")
with open(stats_path) as f:
    stats = json.load(f)

print("Fixing stats.json for GR00T compatibility...")
print()

# Fix action and observation.state count fields
for key in ["action", "observation.state"]:
    if key in stats:
        old_count = stats[key].get("count", [])
        print(f"{key}:")
        print(f"  Old count: {old_count} (shape: {len(old_count)})")

        # Get the dimension from any other stat (like 'mean')
        dimension = len(stats[key].get("mean", []))

        if len(old_count) == 1 and dimension > 1:
            # Replicate the single count value across all dimensions
            new_count = [old_count[0]] * dimension
            stats[key]["count"] = new_count
            print(f"  New count: {new_count} (shape: {len(new_count)})")
            print(f"  ✅ Fixed!")
        else:
            print(f"  Already correct or unexpected format")
        print()

# Backup original
backup_path = stats_path.with_suffix('.json.backup')
print(f"Creating backup: {backup_path}")
with open(backup_path, 'w') as f:
    json.dump(json.load(open(stats_path)), f, indent=2)

# Save fixed stats
print(f"Saving fixed stats to: {stats_path}")
with open(stats_path, 'w') as f:
    json.dump(stats, f, indent=2)

print()
print("✅ Done! stats.json has been fixed.")
print()
print("Verification:")
print(f"  action count shape: {len(stats['action']['count'])}")
print(f"  observation.state count shape: {len(stats['observation.state']['count'])}")
