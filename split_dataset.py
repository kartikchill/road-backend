# split_dataset.py
import random
from pathlib import Path
import shutil
import sys

# ---- CONFIG ----
BASE = Path("/mnt/d/road_infra_analysis/dataset_training/pothole_master")
IMAGES_DIR = BASE / "train" / "images"
LABELS_DIR = BASE / "train" / "labels"
VALID_IMAGES = BASE / "valid" / "images"
VALID_LABELS = BASE / "valid" / "labels"
TEST_IMAGES  = BASE / "test"  / "images"
TEST_LABELS  = BASE / "test"  / "labels"

VALID_FRAC = 0.10   # 10%
TEST_FRAC  = 0.10   # 10%

# ---- Ensure folders exist ----
for p in (VALID_IMAGES, VALID_LABELS, TEST_IMAGES, TEST_LABELS):
    p.mkdir(parents=True, exist_ok=True)

# ---- Gather image files ----
exts = [".jpg", ".jpeg", ".png"]
imgs = [p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in exts]
imgs.sort()
total = len(imgs)
if total == 0:
    print("No images found in", IMAGES_DIR)
    sys.exit(1)

random.seed(42)
random.shuffle(imgs)

n_valid = max(1, int(total * VALID_FRAC))
n_test  = max(1, int(total * TEST_FRAC))

valid_imgs = imgs[:n_valid]
test_imgs  = imgs[n_valid:n_valid + n_test]

moved_valid = moved_test = skipped_no_label = 0

def move_pair(img_path, dst_img_dir, dst_lbl_dir):
    global skipped_no_label
    stem = img_path.stem
    lbl = LABELS_DIR / f"{stem}.txt"
    if not lbl.exists():
        print(f"[WARN] Missing label for: {img_path.name}  → SKIPPING")
        skipped_no_label += 1
        return False
    shutil.move(str(img_path), str(dst_img_dir / img_path.name))
    shutil.move(str(lbl), str(dst_lbl_dir / lbl.name))
    return True

# Move valid
for p in valid_imgs:
    ok = move_pair(p, VALID_IMAGES, VALID_LABELS)
    if ok: moved_valid += 1

# Move test
for p in test_imgs:
    ok = move_pair(p, TEST_IMAGES, TEST_LABELS)
    if ok: moved_test += 1

print()
print("TOTAL IMAGES (initial):", total)
print("VALID moved:", moved_valid)
print("TEST  moved:", moved_test)
print("SKIPPED (no label):", skipped_no_label)
