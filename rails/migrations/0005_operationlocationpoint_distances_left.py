# Generated by Django 5.0.3 on 2024-03-15 03:04

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('rails', '0004_alter_route_end_date'),
    ]

    operations = [
        migrations.AddField(
            model_name='operationlocationpoint',
            name='distances_left',
            field=models.JSONField(blank=True, null=True),
        ),
    ]