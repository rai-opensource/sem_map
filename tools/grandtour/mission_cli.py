#!/usr/bin/env python3
# Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.
"""
CLI for managing Kleinkram GrandTour missions: list, download, and convert bags.

Uses the kleinkram Python SDK. Requires prior authentication via `klein login`.

Download filters to: dlio, alphasense, tf_minimal, tf_model, livox

Usage:
    python mission_cli.py ETH-1
    python mission_cli.py ETH-1 --download --dest ./data
    python mission_cli.py ETH-1 --convert-to-ros2 --dest ./data
    python mission_cli.py --debug   # inspect projects and missions
"""

DEFAULT_FILE_PATTERNS = ["dlio", "tf_minimal", "livox_undist", "zed2i_images"]
# DEFAULT_FILE_PATTERNS = ["zed2i_images"]

import argparse
import json
import subprocess
import sys
from pathlib import Path

import kleinkram


def _get_attr(obj, *keys, default=None):
    """Get attribute from obj (dict or object) trying multiple keys."""
    for key in keys:
        if hasattr(obj, "get") and callable(getattr(obj, "get")):
            val = obj.get(key)
        else:
            val = getattr(obj, key, None) or getattr(obj, key.lower(), None)
        if val is not None:
            return val
    return default


def _get_metadata_short_name(metadata) -> str | None:
    """Extract Short Name from metadata. Kleinkram stores it as {value: ..., type_: STRING}."""
    short_name_obj = None
    if hasattr(metadata, "get"):
        short_name_obj = metadata.get("Short Name") or metadata.get("short_name")
    else:
        short_name_obj = _get_attr(metadata, "Short Name", "short_name")
    if short_name_obj is None:
        return None
    if hasattr(short_name_obj, "get"):
        return short_name_obj.get("value")
    return _get_attr(short_name_obj, "value", "Value")


def get_mission_by_short_name(project_name: str, short_name: str):
    """Find a mission by its Short Name metadata tag."""
    missions = kleinkram.list_missions(project_names=[project_name])
    for mission in missions:
        metadata = _get_attr(mission, "metadata", "Metadata") or {}
        mission_short_name = _get_metadata_short_name(metadata)
        if mission_short_name == short_name:
            return mission
    return None


def list_mission_files(project_name: str, short_name: str) -> None:
    """List all files in the mission with the given short_name."""
    mission = get_mission_by_short_name(project_name, short_name)
    if mission is None:
        print(f"No mission found with short_name '{short_name}' in project '{project_name}'")
        return

    mission_id = _get_attr(mission, "id", "Id")
    mission_name = _get_attr(mission, "name", "Name", default="?")
    print(f"Mission: {mission_name} (id: {mission_id})")
    print("-" * 60)

    files = kleinkram.list_files(
        project_names=[project_name],
        mission_names=[mission_name],
    )
    if not files:
        print("No files found in this mission.")
        return

    for f in files:
        file_name = _get_attr(f, "name", "Name", default="?")
        file_id = _get_attr(f, "id", "Id", default="?")
        size = _get_attr(f, "size", "Size", default="")
        print(f"  {file_name}  (id: {file_id})  {size}")


def _matches_patterns(file_name: str, patterns: list[str]) -> bool:
    """True if file_name matches any pattern. Pattern must be followed by .bag (excludes _imu, _undist etc)."""
    return any(f"{p}.bag" in file_name for p in patterns)


def download_mission_files(
    project_name: str,
    short_name: str,
    dest: str | Path,
    patterns: list[str] | None = None,
) -> None:
    """Download only files matching patterns (dlio, alphasense, tf_minimal, tf_model, livox)."""
    mission = get_mission_by_short_name(project_name, short_name)
    if mission is None:
        print(f"No mission found with short_name '{short_name}' in project '{project_name}'")
        return

    mission_name = _get_attr(mission, "name", "Name", default="?")
    files = kleinkram.list_files(
        project_names=[project_name],
        mission_names=[mission_name],
    )
    pats = patterns or DEFAULT_FILE_PATTERNS
    filtered = [f for f in files if _matches_patterns(_get_attr(f, "name", "Name", default=""), pats)]
    if not filtered:
        print(f"No files matching {pats} in mission {short_name}")
        return

    file_names = [_get_attr(f, "name", "Name", default="") for f in filtered]
    download_dir = Path(dest) / short_name
    download_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(file_names)} files to {download_dir}:")
    for fn in file_names:
        print(f"  {fn}")

    kleinkram.download(
        project_names=[project_name],
        mission_names=[mission_name],
        file_names=file_names,
        dest=download_dir,
        verbose=True,
    )


def convert_bags_to_ros2(
    short_name: str,
    dest: str | Path,
    delete_bags: bool = False,
) -> None:
    """Convert all .bag files in <dest>/<short_name>/ to MCAP using bag_conversion.py, then optionally delete originals."""
    bag_dir = Path(dest) / short_name
    if not bag_dir.is_dir():
        print(f"Directory not found: {bag_dir}")
        print("Download mission files first with --download")
        return

    bag_files = sorted(bag_dir.glob("*.bag"))
    if not bag_files:
        print(f"No .bag files found in {bag_dir}")
        return

    conversion_script = Path(__file__).parent / "bag_conversion.py"
    if not conversion_script.exists():
        print(f"Conversion script not found: {conversion_script}")
        return

    print(f"Converting {len(bag_files)} bag file(s) in {bag_dir}:")
    for bf in bag_files:
        print(f"  {bf.name}")

    # Run bag_conversion.py with --inbag pointing to the directory (BATCH_MODE)
    cmd = [
        sys.executable,
        str(conversion_script),
        "--inbag", str(bag_dir),
    ]
    print(f"\nRunning: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\nConversion failed with exit code {result.returncode}")
        return

    # Verify mcap files were created
    mcap_dir = bag_dir / "mcap"
    mcap_files = sorted(mcap_dir.glob("*.mcap")) if mcap_dir.is_dir() else []

    if not mcap_files:
        print("\nWarning: No .mcap files found after conversion, skipping bag deletion")
        return

    print(f"\nConversion complete. {len(mcap_files)} .mcap file(s) in {mcap_dir}")

    if delete_bags:
        print(f"Deleting {len(bag_files)} original .bag file(s)...")
        for bf in bag_files:
            bf.unlink()
            print(f"  Deleted {bf.name}")
        print("Done.")
    else:
        total_bag_size = sum(bf.stat().st_size for bf in bag_files)
        print(f"\nOriginal .bag files still present ({total_bag_size / 1e9:.1f} GB).")
        print("Use --delete-bags to remove them after conversion.")


def _inspect(obj, depth=0):
    """Convert object to a JSON-serializable representation for debugging."""
    if obj is None:
        return None
    if hasattr(obj, "__dict__") and not hasattr(obj, "get"):
        return {k: _inspect(v, depth + 1) for k, v in vars(obj).items()}
    if hasattr(obj, "get") and callable(getattr(obj, "get")):
        return {k: _inspect(v, depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_inspect(x, depth + 1) for x in obj[: (5 if depth > 1 else 10)]]
    return obj


def debug_kleinkram(project_name: str) -> None:
    """Print projects and missions from Kleinkram for debugging."""
    print("=== Projects ===")
    projects = kleinkram.list_projects()
    for p in projects:
        name = _get_attr(p, "name", "Name", default="?")
        pid = _get_attr(p, "id", "Id", default="?")
        print(f"  {name} (id: {pid})")
    print()
    print(f"=== Missions in project '{project_name}' ===")
    missions = kleinkram.list_missions(project_names=[project_name])
    if not missions:
        print("  (none or project not found)")
        return
    for m in missions[:5]:
        print(json.dumps(_inspect(m), indent=2, default=str))
        print("  ---")
    if len(missions) > 5:
        print(f"  ... and {len(missions) - 5} more")


def list_available_short_names() -> None:
    """Print available short names from grandtour.json."""
    json_path = Path(__file__).parent / "grandtour.json"
    if not json_path.exists():
        print(f"grandtour.json not found at {json_path}")
        return
    with open(json_path) as f:
        data = json.load(f)
    missions = data.get("missions", [])
    print("Available short names from grandtour.json:")
    for m in missions:
        print(f"  {m['short_name']}: {m['name']}")


def main():
    parser = argparse.ArgumentParser(
        description="GrandTour mission CLI: list, download, and convert bag files"
    )
    parser.add_argument(
        "short_name",
        nargs="?",
        help="Mission short name (e.g. ETH-1, SPX-1, GRI-1)",
    )
    parser.add_argument(
        "-p",
        "--project",
        default="GrandTourDataset",
        help="Kleinkram project name (default: GrandTourDataset)",
    )
    parser.add_argument(
        "-l",
        "--list",
        action="store_true",
        help="List available short names from grandtour.json and exit",
    )
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Show projects and missions from Kleinkram (to find correct project/metadata)",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download files (only dlio, alphasense, tf_minimal, tf_model, livox)",
    )
    parser.add_argument(
        "--dest",
        default=".",
        help="Download destination directory (default: current dir)",
    )
    parser.add_argument(
        "--convert-to-ros2",
        action="store_true",
        help="Convert .bag files in <dest>/<short_name>/ to MCAP (ROS2) using bag_conversion.py",
    )
    parser.add_argument(
        "--delete-bags",
        action="store_true",
        help="Delete original .bag files after successful MCAP conversion",
    )
    args = parser.parse_args()

    if args.debug:
        debug_kleinkram(args.project)
        return

    if args.list:
        list_available_short_names()
        return

    if not args.short_name:
        parser.error("short_name is required (or use --list to see available names)")

    if args.download:
        Path(args.dest).mkdir(parents=True, exist_ok=True)
        download_mission_files(args.project, args.short_name, args.dest)

    if args.convert_to_ros2:
        convert_bags_to_ros2(args.short_name, args.dest, delete_bags=args.delete_bags)

    if not args.download and not args.convert_to_ros2:
        list_mission_files(args.project, args.short_name)


if __name__ == "__main__":
    main()
