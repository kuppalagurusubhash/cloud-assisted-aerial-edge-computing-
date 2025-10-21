# # simulator_emulator.py
import cv2, time, math, base64, os
from socketio import Client

# CONFIG
VIDEO_SOURCE = "0"  # 0 for webcam or "video.mp4"
FPS = 10
OUT_FRAME_FOLDER = "frames_in"
SERVER_URL = "http://127.0.0.1:5001"  # server address

os.makedirs(OUT_FRAME_FOLDER, exist_ok=True)
socketio = Client()
socketio.connect(SERVER_URL)

# video capture
if VIDEO_SOURCE == "0":
    cap = cv2.VideoCapture(0)
else:
    if not os.path.exists(VIDEO_SOURCE):
        print(f"⚠️ {VIDEO_SOURCE} not found, using webcam")
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(VIDEO_SOURCE)

if not cap.isOpened():
    raise RuntimeError("Cannot open video source")

# flight path
frame_id = 0
start = time.time()
center_lat = 28.6139
center_lon = 77.2090
radius_m = 50

def offset_lat_lon(lat, lon, dx, dy):
    dlat = dy / 111111.0
    dlon = dx / (111111.0 * math.cos(math.radians(lat)))
    return lat + dlat, lon + dlon

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame_id += 1
        t = time.time() - start
        ang = (t / 30.0) * 2 * math.pi
        dx = radius_m * math.cos(ang)
        dy = radius_m * math.sin(ang)
        lat, lon = offset_lat_lon(center_lat, center_lon, dx, dy)
        alt = 20 + 5*math.sin(ang*2)

        telemetry = {"lat": lat, "lon": lon, "alt": alt, "timestamp": time.time()}

        # frame to base64
        _, buf = cv2.imencode(".jpg", cv2.resize(frame, (960,540)))
        img_b64 = base64.b64encode(buf).decode("utf-8")

        # emit detection
        socketio.emit("detection", {
            "frame_id": frame_id,
            "timestamp": time.time(),
            "telemetry": telemetry,
            "detections": [],
            "image_b64": img_b64
        }, namespace="/")

        # optional display
        cv2.imshow("Simulator", cv2.resize(frame, (960,540)))
        if cv2.waitKey(int(1000/FPS)) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Simulator stopped")

finally:
    cap.release()
    cv2.destroyAllWindows()
    socketio.disconnect()
