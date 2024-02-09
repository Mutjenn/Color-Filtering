import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # görüntüyü hsv dönüştür
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #alt üst hue değelerini belirle
    lower_color = np.array([0, 50, 50])
    upper_color = np.array([22, 255, 255])

    # görüntüyü maskeleme
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # bitwise and kullanarak filtreleme
    res = cv2.bitwise_and(frame, frame, mask=mask)
    
    #contours bul
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    #contours çizimi
    if len(contours)>0:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(frame,[largest_contour],0,(0,255,0),3)


    cv2.imshow("Camera", frame)  # frame değişkenini kullanarak sonucu göster
    cv2.imshow("Filtered Camera", res) # res değişkenini kullanarak sonucu göster
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
