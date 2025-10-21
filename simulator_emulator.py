# simulator_emulator.py
import cv2, time, math, base64, threading, random
from socketio import Client
from ultralytics import YOLO

# ---------------- CONFIG ----------------
VIDEO_SOURCE = 0          # 0 for webcam or path to video
FPS = 10
SERVER_URL = "http://127.0.0.1:5001"
NUM_DRONES = 3
RADIUS_M = 60
CENTER_LAT, CENTER_LON = 12.9716, 77.5946

# Load YOLO model
model = YOLO("yolov8n.pt")

# Shared disaster location
target_location = None
target_lock = threading.Lock()

# ---------------- UTILITY ----------------
def offset_lat_lon(lat, lon, dx, dy):
    """Convert small meter offsets into lat/lon offsets."""
    dlat = dy / 111111.0
    dlon = dx / (111111.0 * math.cos(math.radians(lat)))
    return lat + dlat, lon + dlon

# ---------------- DRONE SIMULATION ----------------
def run_drone(drone_id):
    global target_location
    socketio = Client(reconnection=True)
    socketio.connect(SERVER_URL, namespaces=["/"])
    print(f"✅ {drone_id} connected to server")

    # Receive disaster updates
    @socketio.on("disaster_target", namespace="/")
    def on_target(data):
        global target_location
        with target_lock:
            target_location = data
        print(f"{drone_id} received disaster: {data['type']} at ({data['lat']:.5f},{data['lon']:.5f})")

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source for {drone_id}")

    start_time = time.time()
    frame_id = 0
    FRAME_INTERVAL = 1.0 / FPS

    try:
        while True:
            loop_start = time.time()
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            frame_id += 1
            t = time.time() - start_time
            lat, lon = CENTER_LAT, CENTER_LON

            # Move toward disaster if exists
            with target_lock:
                if target_location:
                    lat_diff = target_location["lat"] - lat
                    lon_diff = target_location["lon"] - lon
                    step = 0.00005
                    lat += step if lat_diff > 0 else -step
                    lon += step if lon_diff > 0 else -step
                    cv2.putText(frame, f"MOVING TO {target_location['type']}", (50,50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 3)
                else:
                    # Circular patrol
                    ang = (t / 30.0 + drone_id * 0.5) * 2 * math.pi
                    dx = RADIUS_M * math.cos(ang)
                    dy = RADIUS_M * math.sin(ang)
                    lat, lon = offset_lat_lon(CENTER_LAT, CENTER_LON, dx, dy)

            alt = 50 + 20 * math.sin(t)
            speed = 10 + 3 * math.cos(t)
            battery = max(10, 100 - (t/60)%100)

            telemetry = {
                "lat": lat, "lon": lon, "alt": alt, "speed": speed,
                "battery": battery, "timestamp": time.time(),
                "drone_id": f"Drone-{drone_id}"
            }

            # YOLO human detection
            detections = []
            results = model(frame)
            for r in results:
                for box in r.boxes:
                    if int(box.cls) == 0:  # class 0 = person
                        detections.append({"class":"person", "confidence": float(box.conf)})

            # Encode frame
            _, buf = cv2.imencode(".jpg", cv2.resize(frame,(960,540)))
            img_b64 = base64.b64encode(buf).decode("utf-8")

            # Send detection to server
            socketio.emit("detection", {
                "frame_id": frame_id,
                "timestamp": telemetry["timestamp"],
                "telemetry": telemetry,
                "detections": detections,
                "image_b64": img_b64
            }, namespace="/")

            # Local preview
            preview = frame.copy()
            cv2.putText(preview, f"{telemetry['drone_id']} Alt:{alt:.1f}m Bat:{battery:.0f}%",
                        (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.imshow(f"{telemetry['drone_id']} Simulator", cv2.resize(preview,(640,360)))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Maintain FPS
            elapsed = time.time() - loop_start
            time.sleep(max(0, FRAME_INTERVAL - elapsed))

    except KeyboardInterrupt:
        print(f"🛑 {drone_id} stopped manually")
    finally:
        cap.release()
        socketio.disconnect()
        cv2.destroyAllWindows()

# ---------------- MAIN ----------------
if __name__ == "__main__":
    print("🚁 Launching multi-drone simulator...")
    threads = []
    for i in range(NUM_DRONES):
        t = threading.Thread(target=run_drone, args=(i+1,))
        t.daemon = True
        t.start()
        threads.append(t)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("🛑 All drone simulators stopped")
