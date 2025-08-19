import os
import sys
import json
import shutil
import cv2
from pathlib import Path
from roboflow import Roboflow
from dotenv import load_dotenv
from collections import Counter
import matplotlib.pyplot as plt

# Load configuration from .env
load_dotenv()
api_key = os.getenv("ROBOFLOW_API_KEY")
model_id_raw = os.getenv("ROBOFLOW_MODEL_ID")
confidence_str = os.getenv("CONFIDENCE", "50")
overlap_str = os.getenv("OVERLAP", "50")
workspace_name = os.getenv("ROBOFLOW_WORKSPACE")

if not api_key:
    raise RuntimeError("ROBOFLOW_API_KEY is not set in .env")

if not model_id_raw or "/" not in model_id_raw:
    raise RuntimeError("ROBOFLOW_MODEL_ID must be like 'project-slug/version', e.g. 'applefinder-im9ep/2'")

project_slug, version_str = model_id_raw.split("/", 1)
try:
    version_num = int(version_str)
except ValueError:
    raise RuntimeError("Version in ROBOFLOW_MODEL_ID must be an integer, e.g. '.../2'")

try:
    confidence_val = int(confidence_str)
    overlap_val = int(overlap_str)
except ValueError:
    raise RuntimeError("CONFIDENCE and OVERLAP must be integers (0-100)")

rf = Roboflow(api_key=api_key)
ws = rf.workspace(workspace_name) if workspace_name else rf.workspace()
project = ws.project(project_slug)
model = project.version(version_num).model

# Ensure outputs directory exists
out_dir = Path("outputs")
out_dir.mkdir(exist_ok=True)

# Load image from command line (fallback to sample.jpg)
if len(sys.argv) >= 2:
    image_path = sys.argv[1]
else:
    default_candidate = Path("sample.jpg")
    if default_candidate.exists():
        print("No image path provided. Using sample.jpg")
        image_path = str(default_candidate)
    else:
        raise SystemExit(
            "No image provided. Pass a path: 'python run_image.py <path_to_image>' or place 'sample.jpg' in the project root."
        )

# Read and ensure minimum HD (720p on shorter side)
img_path_obj = Path(image_path)
abs_path = str(img_path_obj.resolve())
if not img_path_obj.exists():
    raise SystemExit(
        f"Image not found: {image_path}\nAbsolute path: {abs_path}\n"
        "Tip: pass an absolute path or place 'sample.jpg' in the project root."
    )

orig_image = cv2.imread(image_path)
if orig_image is None:
    raise SystemExit(
        f"Cannot read image: {image_path}\nAbsolute path: {abs_path}\n"
        "The file may be corrupted or not a supported format. Try re-saving as JPG/PNG."
    )

h, w = orig_image.shape[:2]
min_side = min(h, w)
processed_path = image_path
draw_image = orig_image

if min_side < 720:
    scale = 720.0 / float(min_side)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(orig_image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    processed_path = str(out_dir / "processed_input.jpg")
    cv2.imwrite(processed_path, resized)
    draw_image = resized
    print(f"Upscaled input to {new_w}x{new_h} (saved to outputs/processed_input.jpg)")
else:
    print(f"Input size OK: {w}x{h}")

# Run prediction
print("Running inference...")
result = model.predict(processed_path, confidence=confidence_val, overlap=overlap_val).json()

# Save raw predictions JSON
pred_json_path = out_dir / "predictions.json"
with open(pred_json_path, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

predictions = result.get("predictions", [])
print(f"Detections: {len(predictions)}")

# Draw bounding boxes locally (on processed image)
image = draw_image
for obj in predictions:
    x, y, w, h = obj['x'], obj['y'], obj['width'], obj['height']
    label = f"{obj['class']} ({obj['confidence']:.2f})"
    pt1 = (int(x - w / 2), int(y - h / 2))
    pt2 = (int(x + w / 2), int(y + h / 2))
    cv2.rectangle(image, pt1, pt2, (0, 255, 0), 2)
    cv2.putText(image, label, (pt1[0], max(0, pt1[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

out_img_path = out_dir / "result.jpg"
cv2.imwrite(str(out_img_path), image)
print(f"Saved annotated image to {out_img_path}")

if not predictions:
    print("No detections. Try lowering CONFIDENCE in .env (e.g., CONFIDENCE=15) and re-run.")
    # Create placeholder charts so docs/ always shows something
    placeholders = [
        ("confidence_hist.png", "Confidence distribution"),
        ("class_counts.png", "Predicted objects count by class"),
    ]
    for filename, title in placeholders:
        plt.figure(figsize=(6, 4))
        plt.text(0.5, 0.5, "No detections", ha="center", va="center", fontsize=14)
        plt.title(title)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(str(out_dir / filename), dpi=150)
        plt.close()
else:
    # Generate confidence histogram
    confidences = [p.get('confidence', 0.0) for p in predictions]
    plt.figure(figsize=(6, 4))
    plt.hist(confidences, bins=10)
    plt.title("Confidence distribution")
    plt.xlabel("Confidence")
    plt.ylabel("Count")
    plt.tight_layout()
    conf_path = out_dir / "confidence_hist.png"
    plt.savefig(str(conf_path), dpi=150)
    plt.close()

    # Generate class counts
    class_counts = Counter(p.get('class', 'unknown') for p in predictions)
    classes = list(class_counts.keys())
    values = list(class_counts.values())
    plt.figure(figsize=(6, 4))
    plt.bar(classes, values)
    plt.title("Predicted objects count by class")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    class_counts_path = out_dir / "class_counts.png"
    plt.savefig(str(class_counts_path), dpi=150)
    plt.close()

    print("Saved charts to outputs/confidence_hist.png and outputs/class_counts.png")

# Also export to docs/ so it's visible on GitHub
docs_dir = Path("docs")
docs_dir.mkdir(exist_ok=True)
try:
    shutil.copy(str(out_img_path), str(docs_dir / "annotated.jpg"))
    print("Copied annotated image to docs/annotated.jpg")
except Exception as e:
    print(f"Warning: could not copy to docs/: {e}")

# Also copy charts and JSON if they exist
for name in ("confidence_hist.png", "class_counts.png", "predictions.json"):
    p = out_dir / name
    if p.exists():
        try:
            shutil.copy(str(p), str(docs_dir / name))
        except Exception as e:
            print(f"Warning: could not copy {name} to docs/: {e}")
