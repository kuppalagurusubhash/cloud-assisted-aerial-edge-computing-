# detector_client.py
import os, time, json, base64, cv2, numpy as np, random
from ultralytics import YOLO
import socketio

# ---------------- CONFIG ---------------- #
FRAME_FOLDER = "frames_in"           # where raw frames are stored
OUT_FOLDER = "frames_out"            # where YOLO overlay frames go
SERVER_URL = "http://127.0.0.1:5001" # Flask-SocketIO server URL
MODEL_NAME = "yolov8n.pt"            # Ultralytics YOLO model
TARGET_CLASSES = ["person"]          # classes to detect
# ---------------------------------------- #

os.makedirs(OUT_FOLDER, exist_ok=True)

# SocketIO client
sio = socketio.Client(reconnection=True)
print(f"🌐 Connecting to {SERVER_URL} ...")
sio.connect(SERVER_URL, namespaces=["/"])
print("✅ Connected to Flask server")

# Load YOLO model
model = YOLO(MODEL_NAME)
print(f"🤖 YOLO model loaded: {MODEL_NAME}")


def get_latest_frames():
    """Get all frames from frames_in sorted by timestamp."""
    files = sorted(
        [f for f in os.listdir(FRAME_FOLDER) if f.endswith(".jpg")],
        key=lambda x: os.path.getmtime(os.path.join(FRAME_FOLDER, x))
    )
    return [os.path.join(FRAME_FOLDER, f) for f in files]


def infer_and_send(frame_path):
    """Run YOLO on frame and send to server."""
    # Extract drone_id and frame_id from filename pattern overlay_<timestamp>_<frame>.jpg
    fname = os.path.basename(frame_path)
    drone_id = "Unknown"
    if "Drone-" in fname:
        parts = fname.split("_")
        drone_id = parts[0].replace("overlay_", "")
    frame_id = int(time.time())

    # Load telemetry if available
    telemetry_path = os.path.splitext(frame_path)[0] + "_telemetry.json"
    if os.path.exists(telemetry_path):
        with open(telemetry_path, "r") as f:
            telemetry = json.load(f)
    else:
        telemetry = {"lat": None, "lon": None, "alt": None, "drone_id": drone_id, "timestamp": time.time()}

    # Read frame
    img = cv2.imread(frame_path)
    if img is None:
        return

    # YOLO detection
    results = model.predict(source=img, conf=0.35, imgsz=640, device="cpu", verbose=False)
    detections = []
    for r in results:
        for b in r.boxes:
            conf = float(b.conf[0])
            cls = int(b.cls[0])
            class_name = model.names[cls]
            if class_name not in TARGET_CLASSES:
                continue
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            detections.append({
                "xyxy": [x1, y1, x2, y2],
                "conf": conf,
                "class": class_name
            })
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{class_name} {conf:.2f}", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Save overlay
    overlay_name = f"{drone_id}_overlay_{int(time.time())}.jpg"
    out_path = os.path.join(OUT_FOLDER, overlay_name)
    cv2.imwrite(out_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])

    # Encode to base64
    with open(out_path, "rb") as f:
        jpg_b64 = base64.b64encode(f.read()).decode("utf-8")

    # Payload for Flask server
    payload = {
        "frame_id": frame_id,
        "timestamp": telemetry.get("timestamp", time.time()),
        "telemetry": telemetry,
        "detections": detections,
        "image_b64": jpg_b64
    }

    # Emit detection event
    sio.emit("detection", payload, namespace="/")
    print(f"📤 Sent frame {frame_id} from {drone_id} ({len(detections)} detections)")

    # Optional alert if a person is found
    if any(d["class"] == "person" for d in detections):
        print(f"⚠️ ALERT: Person detected by {drone_id}!")


if __name__ == "__main__":
    print("🚀 YOLO Detector Client Started...")
    processed = set()

    try:
        while True:
            frames = get_latest_frames()
            for f in frames:
                if f not in processed:
                    infer_and_send(f)
                    processed.add(f)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("🛑 Detector stopped.")
    finally:
        sio.disconnect()
