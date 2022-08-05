import uuid
import cv2
from pyfcm import FCMNotification
from threading import Thread
from ip import get_server_IP

push_service = FCMNotification(
    api_key="AAAAHqEs6HM:APA91bHMPZeO90qsFHOkx-4o0qqrBe0p_pNtu8Fkz1o9zNNt8IhsiJoHJL6afI4PT1KkG92dvDeXfzuciYRRYEVv1Y5DqS97kAgzCZeZ7fuDxh9NtKNrIOokkfsxqjK_OjIvPpMu_VTP")
registration_id = "c3UgrJ7ZSwutJw5cVKaX8a:APA91bEnqQwMxhThb2AykJufQ1dq7ia3qQKEOARWoijK9UqwK12wYn2die_lQTzB3t2j_K8SnEMVZdOGdXP9BPLUvelDVr_Q1N8Wf_MEH443qyNCDVdB5RNP8oW_Ge9RUTDf_1ixoJGx"


def send_notification(title='', body='', img=None):
    img_url = ''

    if img is not None:
        img_path = f'faces/events/{str(uuid.uuid4())}.jpg'
        img_url = f'http://{get_server_IP()}:8000/{img_path}'
        cv2.imwrite(f'mysite/{img_path}', img)

    return push_service.notify_single_device(registration_id=registration_id, message_title=title, message_body=body,
                                             data_message={"img": img_url})


def send_notification_async(title='', body='', img=None):
    t = Thread(target=send_notification, args=(title, body, img))
    print("sent notification")
    t.start()
