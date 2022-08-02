from django.db import models

class Number(models.Model):
    number = models.CharField(max_length=100)

    class Meta:
        ordering = ['-id']
