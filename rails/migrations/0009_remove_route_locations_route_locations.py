# Generated by Django 5.0.3 on 2024-03-15 03:49

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('rails', '0008_remove_operationlocationpoint_distances_left_and_more'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='route',
            name='locations',
        ),
        migrations.AddField(
            model_name='route',
            name='locations',
            field=models.JSONField(blank=True, null=True),
        ),
    ]
