# scenic_server.py
import random
import json
from pathlib import Path
from typing import Optional, List
from multiprocessing import Pool, cpu_count

import scenic
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

SCENARIO_PATH = str((Path(__file__).resolve().parent.parent / "examples" / "gta" / "twoCars.scenic").resolve())

DEFAULT_NUM_SCENES = 500_000
MAX_NUM_SCENES     = 500_000

N_PROCS = min(10, cpu_count())

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    
    
    allow_headers=["*"],
)

# Serve the visualizer and other static assets from the server directory.
server_dir = Path(__file__).resolve().parent
app.mount("/server", StaticFiles(directory=str(server_dir)), name="server_static")


@app.get("/")
def root():
    """Serve the visualizer HTML at the site root."""
    return FileResponse(server_dir / "visualizer.html")

print(f"[INFO] Using up to {N_PROCS} worker processes.")
print(f"[INFO] Scenic scenario path = {SCENARIO_PATH}")
print(cpu_count())

class GenerateRequest(BaseModel):
    num_scenes: int = DEFAULT_NUM_SCENES
    seed: Optional[int] = 0
    scenic_source: Optional[str] = None
    save_to_file: bool = False

def scene_to_record(scene, index: int):
    scene_entry = {
        "scene_index": index,
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

        scene_entry["objects"].append(
            {
                "type": obj_type,
                "x": x,
                "y": y,
                "heading": float(heading) if heading is not None else None,
                "isEgo": is_ego,
            }
        )

    return scene_entry

def worker_task(args):
    wid, n_scenes, seed, scenic_source = args
    print(f"[worker {wid}] generating {n_scenes} scenes (seed={seed})")

    random.seed(seed)

    if scenic_source:
        base_file = str(Path(SCENARIO_PATH).resolve())
        patched_source = f"__file__ = {repr(base_file)}\n" + scenic_source

        scenario = scenic.scenarioFromString(patched_source, mode2D=True)
    else:
        resolved = str(Path(SCENARIO_PATH).resolve())
        print(f"[worker {wid}] loading scenario from: {resolved}")
        try:
            scenario = scenic.scenarioFromFile(resolved, mode2D=True)
        except FileNotFoundError as e:
            print(f"[worker {wid}] ERROR: scenario file not found at {resolved}")
            raise
        
    scenes, num_iters = scenario.generateBatch(n_scenes)
    print(f"[worker {wid}] done (iters={num_iters})")

    records = [scene_to_record(sc, i) for i, sc in enumerate(scenes)]
    return records

@app.get("/scenario_source")
def get_scenario_source():
    path = Path(SCENARIO_PATH).resolve()
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as e:
        return {"path": str(path), "source": "", "error": str(e)}

    return {
        "path": str(path),
        "source": text,
    }


@app.post("/generate")
def generate_scenes(req: GenerateRequest):
    total = min(req.num_scenes, MAX_NUM_SCENES)
    seed  = req.seed if req.seed is not None else 0

    if total <= 0:
        return {"num_scenes": 0, "num_iters": 0, "scenes": []}

    print(f"[INFO] Request: total={total}, seed={seed}, save_to_file={req.save_to_file}")
    if req.scenic_source:
        print("[INFO] Scenic source provided in request (using scenarioFromString).")
    else:
        print(f"[INFO] No scenic_source provided; using default file: {SCENARIO_PATH}")

    procs = min(N_PROCS, total)
    base_chunk = total // procs
    remainder  = total % procs

    tasks = []
    current = 0
    for wid in range(procs):
        n = base_chunk + (1 if wid < remainder else 0)
        if n == 0:
            continue
        w_seed = seed + wid * 9973
        tasks.append((wid, n, w_seed, req.scenic_source))
        current += n

    assert current == total, f"chunk sum {current} != total {total}"

    print(f"[INFO] Spawning {len(tasks)} worker(s) for {total} scenes...")
    with Pool(processes=len(tasks)) as pool:
        parts: List[list] = pool.map(worker_task, tasks)

    all_records = []
    for part in parts:
        all_records.extend(part)

    for idx, rec in enumerate(all_records):
        rec["scene_index"] = idx

    print(f"[INFO] Done: generated {len(all_records)} scenes (requested {total})")

    if req.save_to_file:
        OUTPUT_PATH = Path("twoCars_scenes_from_server.json")
        with OUTPUT_PATH.open("w") as f:
            json.dump(all_records, f)
        print("saved to", OUTPUT_PATH.resolve())

    return {
        "num_scenes": len(all_records),
        "num_iters": -1,
        "scenes": all_records,
    }
