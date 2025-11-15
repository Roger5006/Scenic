#!/usr/bin/env python3
"""Generate many scenes from a Scenic scenario and save results.

Usage examples:
  python tools/generate_scenes.py examples/driving/car.scenic --num 100 --out scenes

This will create the output directory (if needed) and write, for each scene i:
  - scenes/scene_i.bin       : binary encoding from Scenario.sceneToBytes
  - scenes/scene_i.json      : JSON summary containing object positions and params
  - scenes/scene_i.scenic    : (optional) Scenic code reproducing the scene

The script uses Scenario.generateBatch to sample scenes efficiently.
"""

import argparse
import json
import os
from pathlib import Path

try:
    import scenic
except Exception:
    # If running from repo root without installing package, add local src/ to path
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    sys.path.insert(0, str(src_path))
    import scenic

from scenic.core.serialization import scenicToJSON


def _serialize_value(v):
    """Serialize basic Scenic values to JSON-friendly Python types."""
    # Vectors are handled by scenicToJSON
    try:
        return scenicToJSON(v)
    except Exception:
        pass
    # Primitives
    if v is None or isinstance(v, (int, float, str, bool)):
        return v
    # Bytes -> hex
    if isinstance(v, bytes):
        return list(v)
    # Iterables (lists/tuples)
    try:
        if hasattr(v, "__iter__") and not isinstance(v, (str, bytes, dict)):
            return [
                _serialize_value(x)
                for x in v
            ]
    except Exception:
        pass
    # Fallback to repr
    try:
        return repr(v)
    except Exception:
        return str(type(v))


def summarize_scene(scene, scenario=None):
    """Return a JSON-serializable rich summary of a Scene's objects, params, and sampled values.

    If `scenario` is provided, sampled dependency values from `scenario.dependencies`
    that appear in `scene.sample` will be included under the `sampled_values` key.
    """
    summary = {}
    # Global params
    try:
        summary["params"] = {k: _serialize_value(v) for k, v in scene.params.items()}
    except Exception:
        summary["params"] = {}

    # Ego index
    try:
        ego_idx = None
        if scene.egoObject is not None:
            ego_idx = next((i for i, o in enumerate(scene.objects) if o is scene.egoObject), None)
        summary["ego_index"] = ego_idx
    except Exception:
        summary["ego_index"] = None

    # Workspace info (string and any bounds available)
    try:
        summary["workspace"] = repr(scene.workspace)
    except Exception:
        summary["workspace"] = str(type(scene.workspace))

    # detect 2D mode from scenario if possible
    is2d = False
    try:
        if scenario is not None and hasattr(scenario, "compileOptions"):
            is2d = bool(getattr(scenario.compileOptions, "mode2D", False))
    except Exception:
        is2d = False

    objs = []
    for idx, obj in enumerate(scene.objects):
        o = {}
        o["index"] = idx
        o["class"] = obj.__class__.__name__
        # Try position (respect 2D mode)
        pos = None
        try:
            coords = list(obj.position)
            if is2d and len(coords) >= 2:
                coords = coords[:2]
            pos = _serialize_value(coords)
        except Exception:
            try:
                x = getattr(obj.position, "x", None)
                y = getattr(obj.position, "y", None)
                z = getattr(obj.position, "z", None)
                coords = (x, y) if is2d else (x, y, z)
                pos = _serialize_value(coords)
            except Exception:
                pos = None
        o["position"] = pos
        # Heading / orientation
        if hasattr(obj, "heading"):
            try:
                o["heading"] = float(obj.heading)
            except Exception:
                o["heading"] = _serialize_value(obj.heading)
        if hasattr(obj, "orientation"):
            try:
                o["orientation"] = _serialize_value(list(obj.orientation))
            except Exception:
                o["orientation"] = _serialize_value(obj.orientation)
        # Size-related
        for prop in ("width", "length", "height", "radius"):
            if hasattr(obj, prop):
                try:
                    val = getattr(obj, prop)
                    o[prop] = _serialize_value(val)
                except Exception:
                    o[prop] = repr(getattr(obj, prop))
        # Common flags
        for flag in ("allowCollisions", "requireVisible", "occluding", "render"):
            if hasattr(obj, flag):
                try:
                    o[flag] = bool(getattr(obj, flag))
                except Exception:
                    o[flag] = _serialize_value(getattr(obj, flag))
        # Color if present
        if hasattr(obj, "color") and getattr(obj, "color") is not None:
            o["color"] = _serialize_value(getattr(obj, "color"))
        # Sensors names
        if hasattr(obj, "sensors"):
            try:
                o["sensors"] = list(getattr(obj, "sensors").keys())
            except Exception:
                o["sensors"] = _serialize_value(getattr(obj, "sensors"))

        objs.append(o)

    summary["objects"] = objs

    # Sampled dependency values (if scenario and scene.sample available)
    if scenario is not None and hasattr(scenario, "dependencies") and hasattr(scene, "sample"):
        sampled = []
        for dep in scenario.dependencies:
            try:
                if dep in scene.sample:
                    dep_name = getattr(dep, "name", None) or repr(dep)
                    sampled.append({"dep": dep_name, "value": _serialize_value(scene.sample[dep])})
            except Exception:
                # Some dependency types may not be hashable or comparable; skip safely
                try:
                    dep_name = getattr(dep, "name", None) or repr(dep)
                    sampled.append({"dep": dep_name, "value": _serialize_value(getattr(scene, "sample", {}).get(dep, None))})
                except Exception:
                    pass
        summary["sampled_values"] = sampled

    return summary


def main():
    p = argparse.ArgumentParser(description="Generate scenes from a Scenic scenario")
    p.add_argument("scenario", help="Path to a .scenic scenario file")
    p.add_argument("--num", "-n", type=int, default=1, help="Number of scenes to generate")
    p.add_argument("--out", "-o", default="scenes", help="Output directory")
    p.add_argument("--save-scenic", action="store_true", help="Also save Scenic code for each scene")
    p.add_argument("--mode2d", action="store_true", help="Compile scenario in 2D compatibility mode")
    p.add_argument("--max-iterations", type=int, default=2000, help="Max sampling iterations per scene (passed through)")
    p.add_argument("--verbosity", "-v", type=int, default=0, help="Verbosity level")
    p.add_argument("--seed", type=int, default=None, help="Optional RNG seed for reproducibility")
    args = p.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.seed is not None:
        import random

        import numpy as np

        random.seed(args.seed)
        np.random.seed(args.seed)

    print(f"Compiling scenario {args.scenario} (mode2D={args.mode2d})...")
    scenario = scenic.scenarioFromFile(args.scenario, mode2D=args.mode2d)

    print(f"Generating {args.num} scenes...")
    scenes, iterations = scenario.generateBatch(args.num, maxIterations=args.max_iterations, verbosity=args.verbosity)

    print(f"Generated {len(scenes)} scenes (used {iterations} iterations)")

    for i, scene in enumerate(scenes, start=1):
        base = outdir / f"scene_{i:06d}"
        # Binary encoding
        try:
            data = scenario.sceneToBytes(scene)
            with open(base.with_suffix(".bin"), "wb") as f:
                f.write(data)
        except Exception as e:
            print(f"Warning: could not write binary scene for {i}: {e}")

        # JSON summary (positions, params)
        try:
            summary = summarize_scene(scene, scenario=scenario)
            with open(base.with_suffix(".json"), "w") as f:
                json.dump(summary, f, default=scenicToJSON, indent=2)
        except Exception as e:
            print(f"Warning: could not write JSON summary for {i}: {e}")

        # Optional: Scenic code reproducing the scene
        if args.save_scenic:
            try:
                with open(base.with_suffix(".scenic"), "w") as f:
                    scene.dumpAsScenicCode(f)
            except Exception as e:
                print(f"Warning: could not dump Scenic code for {i}: {e}")

    print("Done.")


if __name__ == "__main__":
    main()
