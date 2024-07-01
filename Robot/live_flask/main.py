import cv2
import threading
import queue
from flask import Flask, render_template, Response, request
from datetime import datetime
from gpiozero import Motor

class VideoCapture:
    def __init__(self, name):
        self.cap = cv2.VideoCapture(name)
        self.q = queue.Queue()
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()  
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get()

cam_port = 0
cam = VideoCapture(cam_port)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def gen(camera):
    # Get camera frame
    while True:
        frame = camera.read()
        ret, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(cam), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/picture')
def take_picture():
    frame = cam.read()
    today_date = datetime.now().strftime("%m%d%Y-%H%M%S") 
    cv2.imwrite("stream_photo_" + today_date + ".jpg", frame)
    return "None"

@app.route('/move_robot', methods=['POST'])
def move_robot():
    left = request.json.get('left')
    right = request.json.get('right')
    motor_stanga.forward(left)
    motor_dreapta.forward(right)
    print("left: " + str(left) + " right: " + str(right))
    
    return 'Command received', 200
    
motor_stanga = Motor(forward=22, backward=27, enable=17)
motor_dreapta = Motor(forward=24, backward=23, enable=25)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)