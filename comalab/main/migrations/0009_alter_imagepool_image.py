# Generated by Django 3.2.9 on 2021-11-12 06:01

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0008_auto_20211111_2139'),
    ]

    operations = [
        migrations.AlterField(
            model_name='imagepool',
            name='image',
            field=models.FileField(upload_to='images/'),
        ),
    ]