from pyfcm import FCMNotification

push_service = FCMNotification(api_key="AAAAFWG1PfU:APA91bGeXVosVMCwc72YW-Y8vxc2h1tY0E2pmPPIYiDFM7siKD8xH3-_O8Up7NwCjyVCHJCxl_hmFrN6_Zq96lwbe5Nn3lzvjNbHGS4QKkCA3Oc0NTs_38F0uk6VgVQuA1zJm1kSJ_Ex")
registration_id = "e6pBvaFnTdecMGjXlG1pzU:APA91bHyY2ItjGo60k5isxohnjfAsm4FOrsX5IsFeNR5uIDBp9ClQoOTi5wirgxpzVEZYf2ZvOZ3zNNaBFsKX1ivfCk51Wp431AVRlyHze5R9S6xxkfYKzMATUJC2Yt7bkPnXS3KmEyu"

def send_notification(title='', body=''):
    print("sent")
    return push_service.notify_single_device(registration_id=registration_id, message_title=title, message_body=body)
