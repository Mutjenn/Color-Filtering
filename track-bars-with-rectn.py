import cv2
import numpy as np

# en düşük ve en yüksek HSV değerleri
lower_hsv = np.array([0, 0, 0])
upper_hsv = np.array([179, 255, 255])

#trackbars oluşturma
cv2.namedWindow('Color Tracker', cv2.WINDOW_NORMAL)
cv2.createTrackbar('Hue Min', 'Color Tracker', lower_hsv[0], 179, lambda x: None)
cv2.createTrackbar('Sat Min', 'Color Tracker', lower_hsv[1], 255, lambda x: None)
cv2.createTrackbar('Val Min', 'Color Tracker', lower_hsv[2], 255, lambda x: None)
cv2.createTrackbar('Hue Max', 'Color Tracker', upper_hsv[0], 179, lambda x: None)
cv2.createTrackbar('Sat Max', 'Color Tracker', upper_hsv[1], 255, lambda x: None)
cv2.createTrackbar('Val Max', 'Color Tracker', upper_hsv[2], 255, lambda x: None)


cap = cv2.VideoCapture(0)

while True:
    
    ret, frame = cap.read()

    # görüntüyü hsv dönüştür
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Trackbars
    lower_hsv = np.array([
        cv2.getTrackbarPos('Hue Min', 'Color Tracker'),
        cv2.getTrackbarPos('Sat Min', 'Color Tracker'),
        cv2.getTrackbarPos('Val Min', 'Color Tracker')
    ])

    upper_hsv = np.array([
        cv2.getTrackbarPos('Hue Max', 'Color Tracker'),
        cv2.getTrackbarPos('Sat Max', 'Color Tracker'),
        cv2.getTrackbarPos('Val Max', 'Color Tracker')
    ])
    
    # maskeleme
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    #dikdörtgen çizdirmek için
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        x, y, w, h= cv2.boundingRect(contour)
        area = w*h
        if area>500:
            cv2.rectangle(frame, (x,y), (x+w, y+h),(0,255,0),2)

        
            

    # bitwise and kullanarak filtreleme
    res = cv2.bitwise_and(frame, frame, mask=mask)


    # görüntüyü göster
    cv2.imshow('Color Tracker', res)
    cv2.imshow('Normal',frame)

    # q ile çık
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()