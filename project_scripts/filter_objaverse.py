import objaverse
import trimesh
import numpy as np
import json
import os
import signal

POOL_SIZE   = 300
HARD_N      = 10
EASY_N      = 10
MAX_FACES   = 100000  # increase if high-end machine, decrease if low-end
RAY_TIMEOUT = 30      # seconds before skipping ray cast
OUTPUT_JSON = os.path.expanduser("~/project_scripts/dataset_split.json")

PINNED_UIDS = [
    "3564578cde5c42279ead680df1619e3c",
    "8476c4170df24cf5bbe6967222d1a42d",
]

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError()

def compute_difficulty(glb_path, timeout_secs=RAY_TIMEOUT):
    try:
        mesh = trimesh.load(glb_path, force='mesh')
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        if mesh is None or len(mesh.faces) == 0:
            return None

        if len(mesh.faces) > MAX_FACES:
            depth_ratio = extents[2] / lateral
            return {
                "score": float(0.5 * depth_ratio + 0.5 * 0.5),
                "depth_ratio": float(depth_ratio),
                "occlusion_ratio": float(0.5),
                "n_faces": int(len(mesh.faces))
            }

        bounds  = mesh.bounds
        extents = np.maximum(bounds[1] - bounds[0], 1e-6)
        lateral = max(extents[0], extents[1])
        depth_ratio = extents[2] / lateral

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_secs)
        try:
            ray_origins    = mesh.vertices.copy()
            ray_origins[:, 2] = bounds[1][2] + 1.0
            ray_directions = np.tile([0, 0, -1], (len(ray_origins), 1)).astype(np.float64)
            _, _, index_tri = mesh.ray.intersects_location(
                ray_origins=ray_origins,
                ray_directions=ray_directions
            )
            hit_faces = set(index_tri.tolist())
            occlusion_ratio = 1.0 - (len(hit_faces) / len(mesh.faces))
        except TimeoutError:
            occlusion_ratio = 0.5
        finally:
            signal.alarm(0)

        score = 0.5 * depth_ratio + 0.5 * occlusion_ratio
        return {
            "score":            float(score),
            "depth_ratio":      float(depth_ratio),
            "occlusion_ratio":  float(occlusion_ratio),
            "n_faces":          int(len(mesh.faces))
        }
    except Exception:
        return None


def main():
    print("Loading Objaverse UID list...")
    all_uids = objaverse.load_uids()
    remaining_uids = [u for u in all_uids if u not in PINNED_UIDS]

    np.random.seed(42)
    random_sample  = list(np.random.choice(remaining_uids, size=POOL_SIZE - len(PINNED_UIDS), replace=False))
    sampled_uids   = PINNED_UIDS + random_sample
    print(f"Pool: {len(sampled_uids)} UIDs ({len(PINNED_UIDS)} pinned + {len(random_sample)} random). Downloading...")

    objects = objaverse.load_objects(uids=sampled_uids)
    print(f"Downloaded {len(objects)} objects. Scoring difficulty...")

    scores = []
    for i, (uid, glb_path) in enumerate(objects.items()):
        result = compute_difficulty(glb_path)
        if result is not None:
            scores.append({
                "uid":    uid,
                "glb_path": glb_path,
                "pinned": uid in PINNED_UIDS,
                **result
            })
        if (i + 1) % 10 == 0 or (i + 1) == 1:
            print(f"  Processing {i+1}/{len(objects)}... (scored: {len(scores)})", flush=True)

    pinned_scores = [s for s in scores if s["pinned"]]
    other_scores  = [s for s in scores if not s["pinned"]]
    other_scores.sort(key=lambda x: x["score"], reverse=True)

    hard_set = pinned_scores + other_scores[:max(0, HARD_N - len(pinned_scores))]
    easy_set = other_scores[-EASY_N:]

    print(f"\nTop {HARD_N} HARD objects:")
    for s in hard_set:
        tag = " [PINNED]" if s["pinned"] else ""
        print(f"  {s['uid'][:8]}...  score={s['score']:.3f}  depth_ratio={s['depth_ratio']:.2f}  occlusion={s['occlusion_ratio']:.2f}{tag}")

    print(f"\nTop {EASY_N} EASY objects (control):")
    for s in easy_set:
        print(f"  {s['uid'][:8]}...  score={s['score']:.3f}  depth_ratio={s['depth_ratio']:.2f}  occlusion={s['occlusion_ratio']:.2f}")

    split = {
        "pool_size": POOL_SIZE,
        "hard_set":  hard_set,
        "easy_set":  easy_set,
        "all_scored": scores
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(split, f, indent=2)

    print(f"\nSaved split to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
