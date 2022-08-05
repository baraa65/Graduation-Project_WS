import cv2
from fcm import send_notification_async,send_notification
# from fight_fall import fall_fight_model
# from fire import is_fire
from face2 import match, detect_faces_media, match_without_detection
import urllib.request
import numpy as np
import time

from ip import get_IP
from schedules import get_schedules

URL = f'http://{get_IP()}:8080/shot.jpg'
fps_time = 0
cap = cv2.VideoCapture(0)
face_matches = []
_match = None
res = {
    "face": [],
}

last_stranger_check = time.time()
last_face_check = time.time()
processed_stranger = False
processed_faces = []

people_who_arrived = []
behind_schedule_people = []

while True:
    img_arr = np.array(bytearray(urllib.request.urlopen(URL).read()), dtype=np.uint8)
    img = cv2.imdecode(img_arr, -1)
    # ret, img = cap.read()
    img = cv2.resize(img, (480, 360))

    # _is_fire = is_fire(img)
    # _fall_fight = fall_fight_model().detect(img)
    img, check_face, faces_count, boxes, matches = detect_faces_media(img, res['face'])
    # print(check_face)
    if check_face:
        face_matches = matches

        if 'Stranger' in face_matches:
            if time.time() - last_stranger_check > 30:
                processed_stranger = False

            if not processed_stranger:
                send_notification_async('Face Match', 'Stranger around the house', img)
                processed_stranger = True
                last_stranger_check = time.time()

        notify_faces = []

        for match in face_matches:
            if match != 'Stranger':
                people_who_arrived.append(match)
                if time.time() - last_face_check > 30:
                    processed_faces = []
                if match not in processed_faces:
                    notify_faces.append(match)

        if len(notify_faces) > 0:
            send_notification_async('Face Match', f'{",".join(notify_faces)} has arrived', img)
            last_face_check = time.time()
            for face in notify_faces:
                processed_faces.append(face)

    schedules = get_schedules()

    for s_time, name, is_time_passed in schedules:
        if is_time_passed and name not in people_who_arrived and name not in behind_schedule_people:
            send_notification_async('Behind ٍSchedule', f'{name} did\'t come to house on time')
            behind_schedule_people.append(name)


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
