# Generated by Django 5.0.3 on 2024-03-15 03:06

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('rails', '0005_operationlocationpoint_distances_left'),
    ]

    operations = [
        migrations.AddField(
            model_name='route',
            name='weight',
            field=models.FloatField(blank=True, default=0.0, null=True),
        ),
    ]