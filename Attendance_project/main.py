import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

root_dir = "images/"
train_path = os.path.join(root_dir,'classes')
test_path = os.path.join(root_dir,'test_images')

images = []
class_names = []
all_classes = os.listdir(train_path)

for c in all_classes:
    img = cv2.imread(f"{train_path}/{c}")
    images.append(img)
    class_names.append(os.path.splitext(c)[0])


def find_Encodings(image_data):
    encodings = []
    for img in image_data:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodings.append(encode)
    return encodings

def mark_attendance(name):
    with open('Attendence.csv','r+') as f:
        my_data = f.readlines()
        name_list = []
        for line in my_data:
            entry = line.split(',')
            name_list.append(entry[0])
        if(name not in name_list):
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f"\n{name},{dtString}")

class_encodings = find_Encodings(images)
print("Done Encoding!")

cap = cv2.VideoCapture(0)

while True:
    success,frame = cap.read()
    img = cv2.resize(frame,(0,0),None,0.25,0.25)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    face_locs = face_recognition.face_locations(img)
    encodes = face_recognition.face_encodings(img,face_locs)

    for e,f in zip(encodes,face_locs):
        matches = face_recognition.compare_faces(class_encodings,e)
        distances = face_recognition.face_distance(class_encodings,e)
        match_idx = np.argmin(distances)

        if(matches[match_idx]):
            name = class_names[match_idx]
            y1,x2,y2,x1 = f
            y1,x2,y2,x1 = 4*y1,4*x2,4*y2,4*x1
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            mark_attendence(name)

        cv2.imshow('frame',frame)
        cv2.waitKey(1000)

