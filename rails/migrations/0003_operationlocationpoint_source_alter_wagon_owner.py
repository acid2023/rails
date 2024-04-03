# Generated by Django 5.0.3 on 2024-03-15 02:47

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('rails', '0002_alter_station_road'),
    ]

    operations = [
        migrations.AddField(
            model_name='operationlocationpoint',
            name='source',
            field=models.CharField(blank=True, max_length=512, null=True),
        ),
        migrations.AlterField(
            model_name='wagon',
            name='owner',
            field=models.CharField(blank=True, max_length=512, null=True),
        ),
    ]