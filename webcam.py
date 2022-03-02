import cv2

vid = cv2.VideoCapture(0)

while(True):
    ret,frame = vid.read()
    cv2.imshow("frame",frame)
    font = cv2.FONT_HERSHEY_COMPLEX
    cv2.putText(frame,"Hello There",(0,100),font,2,(255,255,255),3)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()