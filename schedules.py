import datetime
import mysql.connector

mydb = mysql.connector.connect(host="localhost", user="root", password="", database="watcher")

def get_schedules():
    mycursor = mydb.cursor()
    mycursor.execute('''
    SELECT * FROM schedules_schedule 
    LEFT JOIN images_image
    ON schedules_schedule.user_id = images_image.id
    ''')
    myresult = mycursor.fetchall()
    current_time = str(datetime.datetime.now().time())

    return [(x[1], x[4], current_time < x[4]) for x in myresult]
