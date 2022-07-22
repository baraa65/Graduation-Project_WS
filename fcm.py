from pyfcm import FCMNotification

push_service = FCMNotification(api_key="AAAAFWG1PfU:APA91bGeXVosVMCwc72YW-Y8vxc2h1tY0E2pmPPIYiDFM7siKD8xH3-_O8Up7NwCjyVCHJCxl_hmFrN6_Zq96lwbe5Nn3lzvjNbHGS4QKkCA3Oc0NTs_38F0uk6VgVQuA1zJm1kSJ_Ex")
registration_id = "fzz8w1c07-iGlia1bKs9kh:APA91bFtdPrO8A1YYFP2hqm0H5DiHEBb_t8TIn3EZJ2HhZmQCaL-57v3nXWiHomFdt6V0quGeWvehk6Tb1qQ6-JXNmHUevyg6_QgpPs4AshsVbSRfpp6eiQ9SnBnBNh4xYzTLb1ajVa-"

def send_notification(title='', body=''):
    return push_service.notify_single_device(registration_id=registration_id, message_title=title, message_body=body)
