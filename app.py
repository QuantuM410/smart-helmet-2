from flask import Flask, Response, render_template
from utils.video_feed import VideoFeed
import atexit

app = Flask(__name__)
video_feed = VideoFeed()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/yolo_feed')
def yolo_feed():
    return Response(video_feed.generate_yolo_feed(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/depth_feed')
def depth_feed():
    return Response(video_feed.generate_depth_feed(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/proximity_status')
def proximity_status():
    return video_feed.get_proximity_status()

def cleanup():
    video_feed.stop_capture()

atexit.register(cleanup)

if __name__ == '__main__':
    app.run(debug=True, threaded=True)