# Generated by Django 3.1.2 on 2021-10-15 16:10

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0004_auto_20211012_1329'),
    ]

    operations = [
        migrations.AddField(
            model_name='imagepool',
            name='risk_class',
            field=models.CharField(default='', max_length=50),
        ),
        migrations.AlterField(
            model_name='imagepool',
            name='id',
            field=models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID'),
        ),
        migrations.AlterField(
            model_name='imagepool',
            name='image',
            field=models.ImageField(upload_to='images/'),
        ),
        migrations.AlterField(
            model_name='imagepool',
            name='risk',
            field=models.FloatField(default=0.0),
        ),
    ]
