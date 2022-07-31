from fight_fall import fall_fight_model
import cv2
cap = cv2.VideoCapture(r"C:\Users\User\Desktop\fight_test2.mp4")
ffmodel = fall_fight_model()

while cap.isOpened():
    ret, frame = cap.read()
    ret, frame = cap.read()
    # ret, frame = cap.read()

    if frame is None:
        break
    ffmodel.detect(frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()