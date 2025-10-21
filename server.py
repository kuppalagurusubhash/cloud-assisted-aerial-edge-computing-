from flask import Flask, render_template, send_from_directory, request, jsonify
from flask_socketio import SocketIO
import os, base64, time, random, threading

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

OUT_FOLDER = "frames_out"
UPLOAD_FOLDER = "uploads"
os.makedirs(OUT_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

history = []
BASE_LAT, BASE_LON = 12.9716, 77.5946
disaster_target = None

# ---------------- ROUTES ----------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/frames/<path:filename>')
def frames(filename):
    return send_from_directory(OUT_FOLDER, filename)

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    return jsonify({"message": "Video uploaded", "path": filepath})

# ---------------- SIMULATION ----------------
def generate_fake_detection(drone_id):
    ts = time.time()
    frame_id = f"{drone_id}_{int(ts)}"
    lat = BASE_LAT + random.uniform(-0.005, 0.005)
    lon = BASE_LON + random.uniform(-0.005, 0.005)
    alt = random.uniform(20, 120)
    speed = random.uniform(5, 20)
    battery = random.uniform(30, 100)

    detections = []
    if random.random() < 0.05:
        detections.append({"class": "person", "confidence": round(random.uniform(0.7, 0.99), 2)})

    telemetry = {
        "lat": lat, "lon": lon, "alt": alt,
        "speed": speed, "battery": battery,
        "drone_id": drone_id
    }

    return {
        "timestamp": ts,
        "frame_id": frame_id,
        "telemetry": telemetry,
        "detections": detections,
        "img": None
    }

# ---------------- DRONE SIMULATION ----------------
def simulate_drones():
    drone_ids = ["Drone-1", "Drone-2", "Drone-3"]
    while True:
        for drone_id in drone_ids:
            record = generate_fake_detection(drone_id)

            # Move toward disaster if exists
            if disaster_target:
                lat_diff = disaster_target["lat"] - record["telemetry"]["lat"]
                lon_diff = disaster_target["lon"] - record["telemetry"]["lon"]
                record["telemetry"]["lat"] += lat_diff * 0.05
                record["telemetry"]["lon"] += lon_diff * 0.05

            history.append(record)
            if len(history) > 500:
                history.pop(0)

            # Human detection alert if confidence >0.8
            if record["detections"] and any(d['class']=='person' and d['confidence']>0.8 for d in record["detections"]):
                socketio.emit('human_alert', {"drone_id": drone_id})

            socketio.emit('new_detection', record, namespace='/')
        time.sleep(3)

threading.Thread(target=simulate_drones, daemon=True).start()

# ---------------- RANDOM DISASTER GENERATOR ----------------
def random_disaster():
    global disaster_target
    disaster_types = ["Fire", "Flood", "Earthquake", "Storm"]
    while True:
        time.sleep(random.randint(15, 30))  # Random interval
        lat = BASE_LAT + random.uniform(-0.01, 0.01)
        lon = BASE_LON + random.uniform(-0.01, 0.01)
        disaster_type = random.choice(disaster_types)
        severity = random.choice(["Low", "Medium", "High"])
        disaster_target = {"lat": lat, "lon": lon, "type": disaster_type, "severity": severity}
        socketio.emit("disaster_target", disaster_target, namespace='/')
        print(f"🔥 Disaster occurred: {disaster_type} at ({lat:.5f}, {lon:.5f}) Severity: {severity}")

threading.Thread(target=random_disaster, daemon=True).start()

# ---------------- SOCKET EVENT ----------------
@socketio.on('detection', namespace='/')
def handle_detection(data):
    ts = data.get("timestamp", time.time())
    frame_id = data.get("frame_id", f"manual_{int(ts)}")
    telemetry = data.get("telemetry", {})
    detections = data.get("detections", [])

    imgpath = None
    b64 = data.get("image_b64")
    if b64:
        fname = f"overlay_{int(ts)}_{frame_id}.jpg"
        path = os.path.join(OUT_FOLDER, fname)
        with open(path, "wb") as f:
            f.write(base64.b64decode(b64))
        imgpath = f"/frames/{fname}"

    record = {
        "timestamp": ts,
        "frame_id": frame_id,
        "telemetry": telemetry,
        "detections": detections,
        "img": imgpath
    }

    history.append(record)
    if len(history) > 500:
        history.pop(0)

    socketio.emit('new_detection', record, namespace='/')

if __name__ == '__main__':
    print("🚀 Flask SocketIO server running at http://127.0.0.1:5001")
    socketio.run(app, host="0.0.0.0", port=5001, debug=True)
