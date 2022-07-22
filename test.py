import cv2
from fcm import send_notification
from fight_fall import fall_fight_model
from fire import is_fire
from face import get_embedding, match, detect_faces
import urllib.request
import numpy as np
import time

URL = "http://192.168.1.109:8080/shot.jpg"
fps_time = 0

while True:
    img_arr = np.array(bytearray(urllib.request.urlopen(URL).read()), dtype=np.uint8)
    img = cv2.imdecode(img_arr, -1)
    img = cv2.resize(img, (480, 360))

    _match = match(img)
    _is_fire = is_fire(img)
    _fall_fight = fall_fight_model().detect(img)

    res = {
        "face": _match,
        "fire": _is_fire,
        "fall": _fall_fight,
    }

    img, faces_count = detect_faces(img, res['face'])

    # Notifications
    if _match is not None:
        send_notification('Face Match', f'{_match} has arrived')

    if _match is None and faces_count > 0:
        send_notification('Face Match', 'Stranger around the house')

    if _is_fire:
        send_notification('Fire Detected', 'A fire broke out in the house')

    # TODO: fall fight notifications

    # end Notifications

    cv2.putText(img, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow('IPWebcam', img)
    fps_time = time.time()


    q = cv2.waitKey(1)
    if q == ord("q"):
        break

cv2.destroyAllWindows()
