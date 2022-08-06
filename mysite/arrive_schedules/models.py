from django.db import models

class Schedule(models.Model):
    time = models.TimeField()
    user_id = models.IntegerField()

    class Meta:
        ordering = ['-id']
