import face_recognition as fr
import cv2

image_of_martin = fr.load_image_file('image/known/martin.jpg')
martin_face_encoding = fr.face_encodings(image_of_martin)[0]

image_of_charlotte = fr.load_image_file('image/known/martin.jpg')
charlotte_face_encoding = fr.face_encodings(image_of_charlotte)[0]

known_faces_encoding=[
    martin_face_encoding,
    charlotte_face_encoding
]

known_face_names=[
    "Martin",
    "Charlotte"
]

cap = cv2.VideoCapture(0)

while True:
    #load frame
    _,frame = cap.read()

    #find faces
    face_locations = fr.face_locations(frame)
    face_encoding = fr.face_encodings(frame,face_locations)

    for (top,right,bottom,left),face_encoding in zip(face_locations,face_encoding):
        matches = fr.compare_faces(known_faces_encoding,face_encoding)

        name="unknown Person"
        if True in matches:
            name = known_face_names[matches.index(True)]
        cv2.rectangle(frame,(left,top),(right,bottom),(255,255,255))

        cv2.rectangle(frame,(left,bottom-20),(right,bottom),(255,255,255),-1)
        cv2.putText(frame,name,(left+6,bottom-5),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,0))
        #cv2.text(frame,(left+6,bottom-text_height-5),name,fill=(255,255,255,255))

    cv2.imshow("video",frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()