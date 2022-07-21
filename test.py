import cv2

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

    res = {
        "face": match(img),
        "fire": is_fire(img),
        "fall": fall_fight_model().detect(img),
    }

    img = detect_faces(img, res['face'])

    cv2.putText(img, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow('IPWebcam', img)
    fps_time = time.time()


    q = cv2.waitKey(1)
    if q == ord("q"):
        break

cv2.destroyAllWindows()
