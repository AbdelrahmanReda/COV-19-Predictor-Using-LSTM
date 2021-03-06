# Generated by Django 3.1.2 on 2021-07-17 23:53

import datetime
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Egypt',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('Date', models.DateField(default=datetime.date.today)),
                ('cumulative_confirmed_cases', models.IntegerField(default=0)),
                ('confirmed_cases', models.IntegerField(default=0)),
                ('cumulative_recovered_cases', models.IntegerField(default=0)),
                ('recovered_cases', models.IntegerField(default=0)),
                ('cumulative_death_cases', models.IntegerField(default=0)),
                ('death_cases', models.IntegerField(default=0)),
                ('average_temperature', models.FloatField(default=0.0)),
                ('is_occasion', models.BooleanField(default=False)),
            ],
        ),
        migrations.CreateModel(
            name='Uk',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('Date', models.DateField(default=datetime.date.today)),
                ('cumulative_confirmed_cases', models.IntegerField(default=0)),
                ('confirmed_cases', models.IntegerField(default=0)),
                ('cumulative_recovered_cases', models.IntegerField(default=0)),
                ('recovered_cases', models.IntegerField(default=0)),
                ('cumulative_death_cases', models.IntegerField(default=0)),
                ('death_cases', models.IntegerField(default=0)),
            ],
        ),
    ]
