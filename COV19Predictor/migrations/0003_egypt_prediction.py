# Generated by Django 3.1.2 on 2021-07-19 09:22

import datetime
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('COV19Predictor', '0002_metadata'),
    ]

    operations = [
        migrations.CreateModel(
            name='egypt_prediction',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('Date', models.DateField(default=datetime.date.today)),
                ('confirmed_cases', models.IntegerField(default=0)),
                ('recovered_cases', models.IntegerField(default=0)),
                ('death_cases', models.IntegerField(default=0)),
            ],
        ),
    ]
