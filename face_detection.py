#importin open computer vision module(cv2)
import cv2

# opening the dataset and storing it in trained_data variable
trained_data = cv2.CascadeClassifier('data.xml')

#opening webcam with cv2  '0' represets the default camera , you can also pass video file as arguments
webcam = cv2.VideoCapture(0)

while True:
    
    # webcam.read() reads the image from the webcame frame by frame , tf represents wheather the frame captured or not and frame represents the actual frame 
    tf, frame = webcam.read()
    
    # now we are conveting the frame into a black and white image and storing it in bwimage 
    bwimg = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    # here is where the acutal face recogintion is occuring , we are sending the black and white fram and sending it to detectMultiScale() function which returns coordinates with length and breath
    face_coordinate = trained_data.detectMultiScale(bwimg)
    
    # x and y represents the top left coordinates and w and h represents the width and height of the detection 
    for (x, y, w, h) in face_coordinate:
        cv2.rectangle(frame, (x, y), (x+w , y+w), (0,255,0) ,2)

    # imshow() is used the view the frame first arguments is just the name of the tab and frame represents the acutal frame
    cv2.imshow('test one',frame)
    keypressed = cv2.waitKey(1)

    # we are checking for the exit condition 81 represents the ASCII of 'q'  and 113 represents the ASCII of 'Q'
    if keypressed == 81 or keypressed == 113:
        break

webcam.release()



print('ended')
