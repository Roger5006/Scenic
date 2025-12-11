# scenic_server.py
import random
import json
import sys
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import numpy as np
from scipy.spatial import cKDTree
import scenic
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from multiprocessing import Pool, cpu_count

ROOT_DIR = Path(__file__).resolve().parent.parent
EXAMPLES_ROOT = ROOT_DIR / "examples"

SCENARIO_PATHS: Dict[str, Path] = {
    "gta": (EXAMPLES_ROOT / "gta" / "twoCars.scenic").resolve(),
    "vacuum": (EXAMPLES_ROOT / "webots" / "vacuum" / "vacuum_simple.scenic").resolve(),
    "mars": (EXAMPLES_ROOT / "webots" / "mars" / "narrowGoal.scenic").resolve(),
}

DEFAULT_EXAMPLE = "gta"

DEFAULT_NUM_SCENES = 500_000
MAX_NUM_SCENES = 500_000

N_PROCS = min(10, cpu_count())

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

server_dir = Path(__file__).resolve().parent
app.mount("/server", StaticFiles(directory=str(server_dir)), name="server_static")


@app.get("/")
def root():
    return FileResponse(server_dir / "visualizer.html")


print(f"[INFO] Using up to {N_PROCS} worker processes.")
print("[INFO] Scenario paths:")
for k, v in SCENARIO_PATHS.items():
    print(f"   - {k}: {v}")
print(f"[INFO] cpu_count = {cpu_count()}")


def normalize_example(example: Optional[str]) -> str:
    if not example:
        return DEFAULT_EXAMPLE
    ex = example.lower()
    if ex not in SCENARIO_PATHS:
        print(f"[WARN] Unknown example '{example}', falling back to '{DEFAULT_EXAMPLE}'")
        ex = DEFAULT_EXAMPLE
    return ex


def get_scenario_path(example: str) -> Path:
    ex = normalize_example(example)
    return SCENARIO_PATHS[ex]


CURB_TREE: Optional[cKDTree] = None


def init_curb_tree():
    global CURB_TREE
    gta_path = get_scenario_path("gta")
    scenario_dir = gta_path.parent
    path = scenario_dir / "map.npz"
    if not path.exists():
        print(f"[WARN] map.npz not found at {path}")
        CURB_TREE = None
        return

    data = np.load(path, allow_pickle=True)
    if "edges" not in data:
        print("[WARN] edges not found in map.npz")
        CURB_TREE = None
        return

    edges = np.asarray(data["edges"], dtype=float)
    if edges.ndim != 2 or edges.shape[1] != 2:
        print(f"[WARN] edges has unexpected shape {edges.shape}")
        CURB_TREE = None
        return

    CURB_TREE = cKDTree(edges)
    print(f"[INFO] Built curb KDTree with {edges.shape[0]} points (for gta example)")


def dist_to_nearest_curb_world(x: float, y: float) -> Optional[float]:
    if CURB_TREE is None:
        return None
    d, _ = CURB_TREE.query([x, y], k=1)
    return float(d)


init_curb_tree()

class GenerateRequest(BaseModel):
    example: str = DEFAULT_EXAMPLE  # "gta" / "vacuum" / "mars"
    num_scenes: int = DEFAULT_NUM_SCENES
    seed: Optional[int] = 0
    scenic_source: Optional[str] = None
    save_to_file: bool = False


def scene_to_record(scene, index: int, example: str):
    scene_entry = {
        "scene_index": index,
        "example": example,
        "params": scene.params,
        "objects": [],
    }

    for obj in scene.objects:
        obj_type = obj.__class__.__name__
        pos = getattr(obj, "position", None)
        heading = getattr(obj, "heading", None)
        is_ego = (obj is scene.egoObject)

        if pos is not None:
            x = float(getattr(pos, "x", 0.0))
            y = float(getattr(pos, "y", 0.0))
        else:
            x = y = None

        obj_type_lower = obj_type.lower()
        vehicle_keywords = ("car", "vehicle", "bus", "truck", "van", "taxi")
        is_vehicle_like = any(kw in obj_type_lower for kw in vehicle_keywords) or is_ego

        if example == "gta" and is_vehicle_like and (x is not None) and (y is not None):
            dist_to_curb = dist_to_nearest_curb_world(x, y)
        else:
            dist_to_curb = None

        scene_entry["objects"].append(
            {
                "type": obj_type,
                "x": x,
                "y": y,
                "heading": float(heading) if heading is not None else None,
                "isEgo": is_ego,
                "distToNearestCurb": dist_to_curb,
            }
        )
    return scene_entry


def worker_task(args: Tuple[int, int, int, str, Optional[str]]):
    wid, n_scenes, seed, example, scenic_source = args
    example = normalize_example(example)
    scenario_path = get_scenario_path(example)
    scenario_dir = scenario_path.parent

    scenario_dir_str = str(scenario_dir)
    if scenario_dir_str not in sys.path:
        sys.path.insert(0, scenario_dir_str)
        print(f"[worker {wid}] prepended to sys.path: {scenario_dir_str}")

    print(f"[worker {wid}] example={example}, n_scenes={n_scenes}, seed={seed}")
    random.seed(seed)

    if scenic_source:
        base_file = str(scenario_path)
        patched_source = f"__file__ = {repr(base_file)}\n" + scenic_source
        if example == "gta":
            scenario = scenic.scenarioFromString(patched_source, mode2D=True)
        else:
            scenario = scenic.scenarioFromString(patched_source)
    else:
        resolved = str(scenario_path)
        print(f"[worker {wid}] loading scenario from: {resolved}")
        try:
            if example == "gta":
                scenario = scenic.scenarioFromFile(resolved, mode2D=True)
            else:
                scenario = scenic.scenarioFromFile(resolved)
        except FileNotFoundError:
            print(f"[worker {wid}] ERROR: scenario file not found at {resolved}")
            raise

    scenes, num_iters = scenario.generateBatch(n_scenes)
    print(f"[worker {wid}] done (iters={num_iters})")

    records = [scene_to_record(sc, i, example) for i, sc in enumerate(scenes)]
    return records


@app.get("/scenario_source")
def get_scenario_source(example: str = Query(DEFAULT_EXAMPLE, description="gta / vacuum / mars")):
    ex = normalize_example(example)
    path = get_scenario_path(ex)
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as e:
        return {"example": ex, "path": str(path), "source": "", "error": str(e)}

    return {
        "example": ex,
        "path": str(path),
        "source": text,
    }


@app.post("/generate")
def generate_scenes(req: GenerateRequest):
    example = normalize_example(req.example)
    total = min(req.num_scenes, MAX_NUM_SCENES)
    seed = req.seed if req.seed is not None else 0

    if total <= 0:
        return {"example": example, "num_scenes": 0, "num_iters": 0, "scenes": []}

    print(
        f"[INFO] Request: example={example}, total={total}, "
        f"seed={seed}, save_to_file={req.save_to_file}"
    )
    if req.scenic_source:
        print("[INFO] Scenic source provided in request (using scenarioFromString).")
    else:
        scenario_path = get_scenario_path(example)
        print(f"[INFO] No scenic_source provided; using default file: {scenario_path}")

    procs = min(N_PROCS, total)
    base_chunk = total // procs
    remainder = total % procs

    tasks: List[Tuple[int, int, int, str, Optional[str]]] = []
    current = 0
    for wid in range(procs):
        n = base_chunk + (1 if wid < remainder else 0)
        if n == 0:
            continue
        w_seed = seed + wid * 9973
        tasks.append((wid, n, w_seed, example, req.scenic_source))
        current += n

    assert current == total, f"chunk sum {current} != total {total}"

    print(f"[INFO] Spawning {len(tasks)} worker(s) for {total} scenes...")
    with Pool(processes=len(tasks)) as pool:
        parts: List[List[dict]] = pool.map(worker_task, tasks)

    all_records: List[dict] = []
    for part in parts:
        all_records.extend(part)

    for idx, rec in enumerate(all_records):
        rec["scene_index"] = idx

    print(f"[INFO] Done: example={example}, generated {len(all_records)} scenes (requested {total})")

    response_payload = {
        "example": example,
        "num_scenes": len(all_records),
        "num_iters": -1,
        "scenes": all_records,
    }

    output_path = server_dir / f"{example}_scenes_from_server.json"
    try:
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(
                response_payload,
                f,
                ensure_ascii=False,
                indent=2
            )
        print(f"[INFO] Saved JSON to {output_path}")
    except Exception as e:
        print(f"[WARN] Failed to save JSON: {e}")

    return response_payload
