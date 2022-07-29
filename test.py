import cv2
import time
from fcm import send_notification
# # from fight_fall import fall_fight_model
# from fire import is_fire
from face2 import get_embedding, match ,detect_faces_media
# import urllib.request
# import numpy as np
import time
import tensorflow
# URL = "http://192.168.43.166:8080/shot.jpg"
fps_time = 0
sentFire = False
sentFace = False

cap= cv2.VideoCapture(r"C:\Users\User\Desktop\face_test.mp4")
# cap = cv2.VideoCapture(r"C:\Users\User\Pose_Estimation\tf-pose-estimation\images\fall\final\tala2.mp4")

cap = cv2.VideoCapture(0)

_match = None
res = {
    "face": None,
}

while True:
    for i in range(3):
        ret, img = cap.read()
    if img is None:
        break
    # img_arr = np.array(bytearray(urllib.request.urlopen(URL).read()), dtype=np.uint8)
    # img = cv2.imdecode(img_arr, -1)
    img = cv2.resize(img, (480, 360))

    # with tensorflow.device('/GPU:0'):
        # _match = match(img)
    # _is_fire = is_fire(img)
    # # _fall_fight = fall_fight_model().detect(img)

    img,check_face, faces_count = detect_faces_media(img,res['face'])

    if check_face  :
        _match = match(img)


    res = {
        "face": _match,
        # "fire": _is_fire,
        # "fall": _fall_fight,
    }

    print(f"faces count {faces_count}")

    # Notifications
    # if _match is not None and (not sentFace):
    #     send_notification('Face Match', f'{_match} has arrived')
    #     sentFace = True

    # if _match is None and faces_count > 0:
    #     send_notification('Face Match', 'Stranger around the house')

    # if _is_fire and not sentFire:
    #     send_notification('Fire Detected', 'A fire broke out in the house')
    #     sentFire = True

    # # TODO: fall fight notifications

    # # end Notifications
    print(f"FPS : {(1.0 / (time.time() - fps_time))}")
    cv2.putText(img, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow('IPWebcam', img)
    fps_time = time.time()


    q = cv2.waitKey(1)
    if q == ord("q"):
        break

cv2.destroyAllWindows()
