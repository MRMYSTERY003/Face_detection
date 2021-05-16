import cv2

trained_data = cv2.CascadeClassifier('data.xml')

webcam = cv2.VideoCapture(0)

while True:
    tf, frame = webcam.read()

    bwimg = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    face_coordinate = trained_data.detectMultiScale(bwimg)
    for (x, y, w, h) in face_coordinate:
        cv2.rectangle(frame, (x, y), (x+w , y+w), (0,255,0) ,2)

    cv2.imshow('test one',frame)
    keypressed = cv2.waitKey(1)

    if keypressed == 81 or keypressed == 113:
        break

webcam.release()



print('ended')