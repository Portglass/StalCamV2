import cv2
import dli
from imutils import face_utils
import argparse

# cmd : python .\detectVisageVideoV2.py
ap = argparse.ArgumentParser()
ap.add_argument("-f","--effet",required=False,help="Rajouter un effet à la caméra")
args=vars(ap.parse_args())

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("doc/shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    detectedFaces = detector(grayFrame, 1)

    for face in detectedFaces:
        shape = predictor(grayFrame, face)
        shape = face_utils.shape_to_np(shape)
        (xR, yR, w, h) = face_utils.rect_to_bb(face)
        cv2.rectangle(frame, (xR, yR), (xR + w, yR + h), (0, 255, 0), 1)
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

        if args["effet"] == "flou":
            flou = frame[yR:yR+h,xR:xR+w]
            flou = cv2.GaussianBlur(flou, (23,23), 30)
            frame[yR:yR+h, xR:xR+w] = flou


    cv2.imshow("video", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()