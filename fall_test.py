import cv2
from fight_fall import fall_fight_model

cap = cv2.VideoCapture(r"C:\Users\User\Pose_Estimation\tf-pose-estimation\images\fall\final\tala2.mp4")
# cap = cv2.VideoCapture(r"C:\Users\User\Pose_Estimation\tf-pose-estimation\images\punch\mePunching3.mp4")
cap= cv2.VideoCapture(r"C:\Users\User\Desktop\face_test.mp4")
while cap.isOpened():
    ret, frame = cap.read()
    ret, frame = cap.read()
    print("done")
    if frame is None:
        break
    _fall_fight = fall_fight_model().detect(frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()