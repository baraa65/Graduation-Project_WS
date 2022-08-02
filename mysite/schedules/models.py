from django.db import models

class Schedule(models.Model):
    time = models.TimeField()

    class Meta:
        ordering = ['-id']
