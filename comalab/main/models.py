from django.db import models

# Create your models here.
class ImagePool(models.Model):
    image = models.ImageField(upload_to='images/')
    filename = models.CharField(max_length=50)
    risk = models.FloatField(default=0.0)
    heatmap_link = models.CharField(max_length=100, default="")

    def __str__(self):
        return self.filename