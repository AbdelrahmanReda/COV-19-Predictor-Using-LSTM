from datetime import datetime
from statistics import mode
import datetime
from django.db import models



class Egypt(models.Model):

    Date = models.DateField(default=datetime.date.today)
    cumulative_confirmed_cases = models.IntegerField(default=0)
    confirmed_cases = models.IntegerField(default=0)
    cumulative_recovered_cases = models.IntegerField(default=0)
    recovered_cases = models.IntegerField(default=0)
    cumulative_death_cases = models.IntegerField(default=0)
    death_cases = models.IntegerField(default=0)
    average_temperature = models.FloatField(default=0.0)
    is_occasion = models.BooleanField(default=False)



class Uk(models.Model):
    Date = models.DateField(default=datetime.date.today)
    cumulative_confirmed_cases = models.IntegerField(default=0)
    confirmed_cases = models.IntegerField(default=0)
    cumulative_recovered_cases = models.IntegerField(default=0)
    recovered_cases = models.IntegerField(default=0)
    cumulative_death_cases = models.IntegerField(default=0)
    death_cases = models.IntegerField(default=0)

class Uk(models.Model):
    Date = models.DateField(default=datetime.date.today)
    cumulative_confirmed_cases = models.IntegerField(default=0)
    confirmed_cases = models.IntegerField(default=0)
    cumulative_recovered_cases = models.IntegerField(default=0)
    recovered_cases = models.IntegerField(default=0)
    cumulative_death_cases = models.IntegerField(default=0)
    death_cases = models.IntegerField(default=0)

class metaData (models.Model):
    Date = models.DateField('2021-07-10')

class egypt_prediction(models.Model):
    Date = models.DateField(default=datetime.date.today)
    confirmed_cases = models.IntegerField(default=0)
    recovered_cases = models.IntegerField(default=0)
    death_cases = models.IntegerField(default=0)
