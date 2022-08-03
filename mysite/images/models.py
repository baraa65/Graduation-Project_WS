from django.db import models

class Image(models.Model):
    name = models.CharField(max_length=100)
    number = models.CharField(max_length=100)
    image = models.ImageField(null=True, blank=True, upload_to='faces/')

    class Meta:
        ordering = ['-id']
