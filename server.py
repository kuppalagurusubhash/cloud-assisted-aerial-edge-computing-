from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO, emit
import os, base64, time

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

OUT_FOLDER = "frames_out"
os.makedirs(OUT_FOLDER, exist_ok=True)
history = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/frames/<path:filename>')
def frames(filename):
    return send_from_directory(OUT_FOLDER, filename)

@socketio.on('detection', namespace='/')
def handle_detection(data):
    frame_id = data.get("frame_id", None)
    ts = data.get("timestamp", time.time())
    telemetry = data.get("telemetry", {})
    detections = data.get("detections", [])

    imgpath = None
    b64 = data.get("image_b64", None)
    if b64:
        fname = f"overlay_{int(ts)}_{frame_id}.jpg"
        path = os.path.join(OUT_FOLDER, fname)
        with open(path, "wb") as f:
            f.write(base64.b64decode(b64))
        imgpath = f"/frames/{fname}"

    record = {"ts": ts, "frame_id": frame_id, "telemetry": telemetry,
              "detections": detections, "img": imgpath}
    history.append(record)
    if len(history) > 500:
        history.pop(0)

    emit('new_detection', record, broadcast=True, namespace='/')
    print(f"Broadcasted frame {frame_id}")

if __name__ == '__main__':
    socketio.run(app, host="0.0.0.0", port=5001, debug=True)
