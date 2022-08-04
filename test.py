import cv2
from fcm import send_notification_async,send_notification
# from fight_fall import fall_fight_model
# from fire import is_fire
from face2 import match, detect_faces_media, match_without_detection
import urllib.request
import numpy as np
import time

from ip import get_IP

URL = f'http://{get_IP()}:8080/shot.jpg'
fps_time = 0
cap = cv2.VideoCapture(0)
face_matches = None
_match = None
res = {
    "face": [],
}

while True:
    img_arr = np.array(bytearray(urllib.request.urlopen(URL).read()), dtype=np.uint8)
    img = cv2.imdecode(img_arr, -1)
    # ret, img = cap.read()
    img = cv2.resize(img, (480, 360))

    # _is_fire = is_fire(img)
    # _fall_fight = fall_fight_model().detect(img)
    img, check_face, faces_count, boxes = detect_faces_media(img, res['face'])
    # print(check_face)
    if check_face:
        img, face_matches = match_without_detection(img, boxes)


    # print(face_matches)
    res = {
        "face": face_matches,
        # "fire": _is_fire,
        # "fall": _fall_fight,
    }


    # img, faces_count = detect_faces_media(img, res['face'])
    # img, faces_count = detect_faces_media(img, '')

    # Notifications
    if _match is not None:
        # send_notification('Face Match', f'{_match} has arrived')
        break
    #
    # if _match is None and faces_count > 0:
    #     send_notification('Face Match', 'Stranger around the house')
    #     break
    # # if _is_fire:
    #     send_notification('Fire Detected', 'A fire broke out in the house')

    # TODO: fall fight notifications

    # end Notifications

    cv2.putText(img, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),
                2)
    cv2.imshow('IPWebcam', img)
    fps_time = time.time()

    q = cv2.waitKey(1)
    if q == ord("q"):
        break

cv2.destroyAllWindows()
