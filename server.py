from flask import Flask, jsonify,request,render_template,Response
from flask_restful import Api, Resource, reqparse
import pickle
import numpy as np
import cv2


# create an api object and a flask app object

global rec_frame,switch ,rec

switch=1
rec=0

app = Flask(__name__, template_folder='./template')
model=pickle.load(open('logmodel50.pkl','rb'))
camera = cv2.VideoCapture(0)
rec= not rec

emoji_dist={0:"image/angry.png",1:"image/disgusted.png",2:"image/fearful.png",3:"image/happy.png",4:"image/neutral.png",5:"image/sad.png",6:"image/surpriced.png"}
show_text=[0]
def gen_frames():  # generate frame by frame from camera
    global out, capture,rec_frame
    while True:
        success, frame = camera.read() 
        
        if success:
            
            if(rec):
                rec_frame=frame
                emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Natural", 5: "Sad", 6: "Surprised"}
                bounding_box = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                num_faces = bounding_box.detectMultiScale(gray_frame,scaleFactor=1.3, minNeighbors=5)

                for (x,y, w, h) in num_faces:
                    cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
                    roi_gray_frame = gray_frame[y:y + h, x:x + w]
                    cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
                    emotion_prediction = model.predict(cropped_img)
                    maxindex = int(np.argmax(emotion_prediction))
                    frame= cv2.putText(cv2.flip(frame,1),emotion_dict[maxindex], (x+20,y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2,cv2.LINE_AA)
                    frame=cv2.flip(frame,1)
                    show_text[0]=maxindex
            
            try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame,1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass
                
        else:
            pass
def create_emoji():
    global show_text
    while True:
        try:
            img=cv2.imread(emoji_dist[show_text[0]])
            ret, buffer = cv2.imencode('.jpg', cv2.flip(img,1))
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            pass


@app.route('/')
def index():
    return render_template('index.html')
    
    
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/emoji')
def emoji():
    return Response(create_emoji(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/requests',methods=['POST','GET'])
def tasks():
    global switch,camera,rec
    if request.method == 'POST':
        
        if  request.form.get('stop') == 'Stop/Start':
            
            if(switch==1):
                switch=0
                camera.release()
                cv2.destroyAllWindows()
                rec=not rec
                
            else:
                camera = cv2.VideoCapture(0)
                switch=1
                
                rec= not rec
                          
                 
    elif request.method=='GET':
        return render_template('index.html')
    return render_template('index.html')


if __name__ == '__main__':
    app.run()
    
camera.release()
cv2.destroyAllWindows() 