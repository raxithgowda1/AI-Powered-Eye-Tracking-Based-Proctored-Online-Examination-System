from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit 
import cv2
import threading
import time
from threading import Lock
from eye_controller import EyeController
from warning_system import WarningSystem
import dlib
from imutils import face_utils

app = Flask(__name__, template_folder='templates')
socketio = SocketIO(app, async_mode='threading')

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW) 
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

eye_controller = EyeController()
warning_system = WarningSystem()

current_mode = None
stop_flag = threading.Event()
frame_lock = Lock()

def generate_frames():
    global current_mode
    
    while not stop_flag.is_set():
        try:
            success = False
            with frame_lock:
                success, frame = camera.read()
                
            if not success:
                print("Failed to get frame from camera")
                time.sleep(0.1)  
                continue
                    
            processed_frame = frame.copy()  
            warning = False
            
            if current_mode == 'eye_control':
                processed_frame = eye_controller.process_frame(processed_frame)
            elif current_mode == 'warning':
                processed_frame, warning = warning_system.process_frame(processed_frame)
                if warning:
                    socketio.emit('warning', {'count': warning_system.warning_count})
            
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            if not ret:
                continue
            
            frame_bytes = buffer.tobytes()
                
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                   
        except Exception as e:
            print(f"Frame generation error: {e}")
            time.sleep(0.1)  

@app.route('/')
def index():
    return render_template('unified.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('connect')
def handle_connect():
    if current_mode is not None:
        emit('mode_update', {'mode': current_mode})

@socketio.on('set_mode')
def handle_mode_change(data):
    global current_mode
    
    new_mode = data['mode']
    print(f"Changing mode to: {new_mode}")
    
    if new_mode == current_mode:
        return
        
    with frame_lock:
        current_mode = new_mode
        
        if new_mode == 'eye_control':
            warning_system.reset()
            eye_controller.activate()
            warning_system.deactivate()
        elif new_mode == 'warning':
            eye_controller.deactivate()
            warning_system.activate()
        else:
            eye_controller.deactivate()
            warning_system.deactivate()
            
    emit('mode_update', {'mode': current_mode}, broadcast=True)

def cleanup():
    global stop_flag
    stop_flag.set()
    time.sleep(0.5) 
    with frame_lock:
        if camera.isOpened():
            camera.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        socketio.run(app, debug=True, use_reloader=False, host='0.0.0.0')
    except KeyboardInterrupt:
        pass
    finally:
        cleanup()