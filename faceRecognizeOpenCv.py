import cv2
import numpy as np
import face_recognition
import os

#######################
path = "faces"
scale = 0.25
box_multiplier = 1/scale
#######################

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)
        print(encode)
        encodeList.append(encode)
    return encodeList

images = []
classNames = []

for img in os.listdir(path):
    image = cv2.imread(path+'/'+img)
    images.append(image)
    classNames.append(os.path.splitext(img)[0])

#name = input("Enter name: ")

knownEncodes = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)

while True:
    sucess,frame = cap.read()

    Current_image = cv2.resize(frame, (0, 0), None, scale, scale)
    Current_image = cv2.cvtColor(Current_image, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(Current_image, model='cnn')
    face_encodes = face_recognition.face_encodings(Current_image, face_locations)

    for encodeFace, faceLocation in zip(face_encodes, face_locations):
        matches = face_recognition.compare_faces(knownEncodes, encodeFace, tolerance=0.6)
        faceDis = face_recognition.face_distance(knownEncodes, encodeFace)
        matchIndex = np.argmin(faceDis)

        # If match found then get the class name for the corresponding match
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
        else:
            name = 'Unknown'

        y1, x2, y2, x1 = faceLocation
        y1, x2, y2, x1 = int(y1 * box_multiplier), int(x2 * box_multiplier), int(y2 * box_multiplier),
        int(x1 * box_multiplier)
        # Draw rectangle around detected face
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img, (x1, y2 - 20), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2)
    cv2.imshow
    if cv2.waitKey(1) == ord('c'):
        filename = 'faces/'+name+'.jpg'
        cv2.imwrite(filename,frame)
        print('image Saved- ',filename)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
