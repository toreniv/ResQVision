import json
import shutil
import pathlib

DRAFT_DIR = pathlib.Path("dataset_draft")
MANIFEST_PATH = DRAFT_DIR / "manifest.json"
REVIEW_CANDIDATES_DIR = DRAFT_DIR / "review_candidates"

if not MANIFEST_PATH.exists():
    print("No manifest found.")
    raise SystemExit(1)

manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

REVIEW_CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)

# Ranking logic:
# High priority: detected > 0 and incomplete_labels == False
# Medium priority: detected > 0 and incomplete_labels == True
# Low priority: detected == 0

candidates = []
for stem, info in manifest.items():
    det = info.get("detected_count", 0)
    inc = info.get("incomplete_labels", False)
    
    if det > 0 and not inc:
        priority = "high"
        score = 3
    elif det > 0 and inc:
        priority = "medium"
        score = 2
    else:
        priority = "low"
        score = 1
        
    info["review_priority"] = priority
    candidates.append({
        "stem": stem,
        "score": score,
        "det": det,
        "priority": priority
    })

# Sort by score descending, then by det descending
candidates.sort(key=lambda x: (x["score"], x["det"]), reverse=True)

for rank, c in enumerate(candidates, start=1):
    stem = c["stem"]
    priority = c["priority"]
    
    # Copy preview image
    src_preview = DRAFT_DIR / "details" / stem / "detection_preview.jpg"
    if src_preview.exists():
        dst_preview = REVIEW_CANDIDATES_DIR / f"{rank:03d}_{priority}_{stem}.jpg"
        shutil.copy2(src_preview, dst_preview)

MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(f"Ranked {len(candidates)} candidates and copied previews to {REVIEW_CANDIDATES_DIR}.")
print("Run python scripts/review_dataset.py to officially approve them.")
