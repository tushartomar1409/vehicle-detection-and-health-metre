from flask import Flask, render_template,url_for,request,session,logging,redirect,flash,send_file
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import create_engine,update
from sqlalchemy.orm import scoped_session,sessionmaker
import os
import datetime
import pandas as pd
from flask_mysqldb import MySQL
import cv2
import numpy as np
import time



#establishing the sql connection
app = Flask(__name__)
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'vdcsystem'

mysql = MySQL(app)

app.config['SECRET_KEY'] = 'secret key'    

def center_handle(x,y,w,h):
    x1=int(w/2)
    y1=int(h/2)
    cx=x+x1
    cy=y+y1
    return cx,cy


def vehicle_counter(video_path):
    cap = cv2.VideoCapture(video_path)

    min_width_rectangle = 80
    min_height_rectangle = 80
    min_area = 1000

    count_line_position = 550


    algo = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)


    def center_handle(x, y, w, h):
        x1 = int(w/2)
        y1 = int(h/2)
        cx = x+x1
        cy = y+y1
        return cx, cy


    detect = []
    offset = 6
    counter = 0
    vehicle_speed = {}
    last_position = {}

    while True:
        ret, video = cap.read()
        gray = cv2.cvtColor(video, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 5)

        vid_sub = algo.apply(blur)
        dilat = cv2.dilate(vid_sub, np.ones((5, 5)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
        dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)
        countersahpe, h = cv2.findContours(
            dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        cv2.line(video, (25, count_line_position),
                (1200, count_line_position), (255, 0, 0), 3)

        for (i, c) in enumerate(countersahpe):
            (x, y, w, h) = cv2.boundingRect(c)
            val_counter = (w >= min_width_rectangle) and (
                h >= min_height_rectangle)
            if not val_counter:
                continue
            cv2.rectangle(video, (x, y), (x+w, y+h), (0, 255, 255), 2)

            center = center_handle(x, y, w, h)
            detect.append(center)
       # cv2.circle(video, center, 4, (0,0,255), -1)

            for (x, y) in detect:
                if y < (count_line_position + offset) and y > (count_line_position - offset):
                    counter += 1
                    cv2.line(video, (25, count_line_position),
                            (1200, count_line_position), (0, 127, 255), 3)
                    detect.remove((x, y))

                    print("Vehicle No: " + str(counter))

                else:
                    time_diff = time.time() - last_position.get((640, 360), 0)

                    if time_diff > 0.5:  # wait for at least 0.5 seconds before counting again
                        last_position[(x, y)] = time.time()
                        if (x, y) in vehicle_speed:
                            speed = (count_line_position-y) / time_diff
                            vehicle_speed[(x, y)].append(speed)
                        else:
                            vehicle_speed[(x, y)] = []
                            vehicle_speed[(x, y)].append(
                                (count_line_position-y) / time_diff)

                        y_offset = 0
            for (x, y) in last_position:
                if (x, y) in vehicle_speed:
                    avg_speed = np.mean(vehicle_speed[(x, y)])
                    cv2.putText(video, "Speed: "+str(round(avg_speed, 2))+" km/h",
                                (x, y-20-y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 244, 0), 2)
                    y_offset += 2

        for contour in countersahpe:

            if cv2.contourArea(contour) > min_area:

                x, y, w, h = cv2.boundingRect(contour)

                aspect_ratio = float(w) / h
                if aspect_ratio < 1.1:
                    label = "Bike"
                elif aspect_ratio > 1.5:
                    label = "car"
                else:
                    continue
            cv2.rectangle(video, (x, y), (x + w, y + h), (0, 255, 100), 1)
            cv2.putText(video, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.putText(video, "Vehicle Count: " + str(counter), (450, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

        cv2.imshow('Detector', video)

        if cv2.waitKey(15) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


@app.route("/login", methods = ['GET', 'POST'])
def login():
     
     username = request.form.get('username')
     password = request.form.get('password')
     
     if request.method =="POST":   
        cur = mysql.connection.cursor()
        
        cur.execute('SELECT username FROM login WHERE username=%s', [username])
        username = cur.fetchone()
        cur.execute("SELECT password FROM login WHERE password=%s", [password])
        password = cur.fetchone()
        print(username)
        print(password)
        
        if username != None and password != None:
            #user = username
            return redirect("home")
        else:
            return redirect("login")
        
        mysql.connection.commit()
        cur.close()
     return render_template('login.html')




@app.route("/home", methods = ['GET', 'POST'])
def home():
    
    video_path = request.form.get("file")
    if request.method == "POST":
        vehicle_counter(video_path)

    return render_template('home.html')



@app.route("/contactus", methods = ['GET', 'POST'])
def contactus():
    
   

    return render_template('contact_us.html')


@app.route("/about", methods = ['GET', 'POST'])
def about():
    
   

    return render_template('about.html')


app.run(debug=True)
