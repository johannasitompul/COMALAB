# Generated by Django 3.2.9 on 2021-11-11 13:39

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0007_imagepool_heatmap'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='imagepool',
            name='heatmap',
        ),
        migrations.AddField(
            model_name='imagepool',
            name='heatmap_link',
            field=models.CharField(default='', max_length=100),
        ),
    ]
