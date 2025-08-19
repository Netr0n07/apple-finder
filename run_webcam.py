import os
import cv2
from pathlib import Path
from roboflow import Roboflow
from dotenv import load_dotenv

# Load configuration
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
version_num = int(version_str)
confidence_val = int(confidence_str)
overlap_val = int(overlap_str)

rf = Roboflow(api_key=api_key)
ws = rf.workspace(workspace_name) if workspace_name else rf.workspace()
project = ws.project(project_slug)
model = project.version(version_num).model

out_dir = Path("outputs")
out_dir.mkdir(exist_ok=True)

cap = cv2.VideoCapture(0)
# Try to enforce 1280x720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Ensure minimum 720p on shorter side
    h, w = frame.shape[:2]
    min_side = min(h, w)
    if min_side < 720:
        scale = 720.0 / float(min_side)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        frame_proc = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        frame_proc = frame

    # Run inference
    result = model.predict(frame_proc, confidence=confidence_val, overlap=overlap_val).json()

    # Draw bounding boxes (scale back if resized)
    scale_back_x = w / frame_proc.shape[1]
    scale_back_y = h / frame_proc.shape[0]

    for obj in result.get('predictions', []):
        x, y, bw, bh = obj['x'], obj['y'], obj['width'], obj['height']
        x *= scale_back_x
        y *= scale_back_y
        bw *= scale_back_x
        bh *= scale_back_y
        label = f"{obj['class']} ({obj['confidence']:.2f})"
        cv2.rectangle(frame, (int(x-bw/2), int(y-bh/2)), (int(x+bw/2), int(y+bh/2)), (0,255,0), 2)
        cv2.putText(frame, label, (int(x), int(y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    cv2.imshow("AppleFinder", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
