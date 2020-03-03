from django.db import models

class Image(models.Model):

    file = models.ImageField(upload_to='./pic', blank=False, null=False)
