# detector_client.py
import os, time, json, base64
from ultralytics import YOLO
import cv2
import numpy as np
import socketio

FRAME_FOLDER = "frames_in"
OUT_FOLDER = "frames_out"
TELEMETRY_FILE = "telemetry.json"
SERVER_URL = "http://localhost:5000"  # change if server remote
MODEL_NAME = "yolov8n.pt"   # ultralytics model; download automatically on first run
PERSON_CLASS_NAMES = ["person"]  # YOLO typical

os.makedirs(OUT_FOLDER, exist_ok=True)

# socketio client
sio = socketio.Client(logger=False, engineio_logger=False)
print("Connecting to server...", SERVER_URL)
sio.connect(SERVER_URL, namespaces=["/"])
print("Connected")

model = YOLO(MODEL_NAME)
print("Model loaded:", MODEL_NAME)

def latest_frame_path():
    files = sorted([f for f in os.listdir(FRAME_FOLDER) if f.endswith(".jpg")])
    return os.path.join(FRAME_FOLDER, files[-1]) if files else None

last_sent = 0
try:
    while True:
        frame_path = latest_frame_path()
        if frame_path is None:
            time.sleep(0.1)
            continue

        # only process new frames
        fid = int(os.path.splitext(os.path.basename(frame_path))[0].split("_")[1])
        if fid <= last_sent:
            time.sleep(0.05)
            continue
        last_sent = fid

        # read telemetry
        telemetry = {}
        try:
            with open(TELEMETRY_FILE, "r") as f:
                telemetry = json.load(f)
        except:
            telemetry = {"lat":None,"lon":None,"alt":None,"timestamp":time.time()}

        img = cv2.imread(frame_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = model.predict(source=img_rgb, conf=0.35, imgsz=640, device="cpu")  # set device appropriately

        detections = []
        # parse results: ultralytics returns a list of Results
        for r in results:
            boxes = getattr(r, "boxes", None)
            if boxes is None: continue
            for b in boxes:
                conf = float(b.conf[0])
                cls = int(b.cls[0])
                # only person class (COCO id 0 usually)
                if cls != 0:
                    continue
                x1,y1,x2,y2 = map(int, b.xyxy[0].tolist())
                detections.append({"xyxy":[x1,y1,x2,y2],"conf":conf,"class":"person"})

                # draw box on img
                cv2.rectangle(img, (x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(img, f"person {conf:.2f}", (x1,y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1)

        # save overlay frame to OUT_FOLDER
        outpath = os.path.join(OUT_FOLDER, f"overlay_{fid:06d}.jpg")
        cv2.imwrite(outpath, img, [int(cv2.IMWRITE_JPEG_QUALITY),85])

        # encode overlay as base64 to emit
        with open(outpath, "rb") as f:
            jpg_b64 = base64.b64encode(f.read()).decode('utf-8')

        payload = {
            "frame_id": fid,
            "timestamp": telemetry.get("timestamp", time.time()),
            "telemetry": telemetry,
            "detections": detections,
            "image_b64": jpg_b64
        }
        # emit over socket.io
        sio.emit("detection", payload, namespace="/")
        print(f"Sent frame {fid} detections: {len(detections)}")

        time.sleep(0.05)

except KeyboardInterrupt:
    print("Exiting detector")
finally:
    try:
        sio.disconnect()
    except:
        pass
