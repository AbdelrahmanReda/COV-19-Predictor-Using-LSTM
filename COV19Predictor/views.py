import datetime
import os
import time

from django.core import serializers
from django.db import connection
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
import math
from datetime import timedelta, datetime


def saveCounry(request):
    countries = ['England']
    for country in countries:
        query = "create table if not exists public." + country + " ( id serial not null constraint \"COV19Predictor_" + country + "_pkey\" primary key,\"Date\" date not null,cumulative_confirmed_cases integer not null,confirmed_cases integer not null,cumulative_recovered_cases integer not null,recovered_cases integer not null,cumulative_death_cases integer not null,death_cases integer not null);alter table public." + country + " owner to postgres;"
        cursor = connection.cursor()
        cursor.execute(query)

    return HttpResponse("Hi :D ")


# def saveCounry(request):
#     cursor = connection.cursor()
#     print(cursor)
#
#     cursor.execute("SELECT * FROM public.\"COV19Predictor_egypt\"")
#     row = cursor.fetchall()
#     print(row)
#     return HttpResponse ("Hi :D ")

# def saveCounry(request):
#     Data = getCountryStats()
#
#     # eg = Egypt(
#     #     Date = '2020-2-1',
#     #     cumulative_confirmed_cases=1,
#     #     confirmed_cases=1,
#     #     cumulative_recovered_cases=1,
#     #     recovered_cases= 1,
#     #     cumulative_death_cases=1,
#     #     death_cases=1,
#     #     average_temperature=0.5,
#     #     is_occasion=False
#     #     )
#     # eg.save()
#     return HttpResponse("Hi")


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import tensorflow as tf;

from .models import Egypt
from .models import Uk
from .models import metaData


def getPrediction(request):
    return render(request, 'Hello.html');


country = "Egypt"
df_confirmed = pd.read_csv(
    "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv")
# df_confirmed.to_csv('global.csv')
df_confirmed_country = df_confirmed[df_confirmed["Country/Region"] == country]
df_confirmed_country = pd.DataFrame(df_confirmed_country[df_confirmed_country.columns[4:]].sum(), columns=["confirmed"])
df_confirmed_country.index = pd.to_datetime(df_confirmed_country.index, format='%m/%d/%y')


def getCountryStats():
    df_confirmed = pd.read_csv(
        "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv")
    df_deaths = pd.read_csv(
        "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv")
    df_recovred = pd.read_csv(
        "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv")
    return [df_confirmed, df_deaths, df_recovred]


def getLst():
    return 0


import itertools as it


def find_ranges(lst, n=4 , number = 1):
    """Return ranges for `n` or more repeated values."""
    groups = ((k, tuple(g)) for k, g in it.groupby(enumerate(lst), lambda x: x[-1]))
    repeated = (idx_g for k, idx_g in groups if len(idx_g) >=n)
    rebeats = ((sub[0][0], sub[-1][0]) for sub in repeated)
    one_rebeats = []
    for r in rebeats:
        if lst[r[0]]==1:
            one_rebeats.append(r)







    return one_rebeats








def adjustForecast(unadjusted_forecast,adjusted_helper):

    start_inx= list(find_ranges(adjusted_helper['predictions'], 5))[0][0]
    print('i is ',start_inx)
    for i in range (start_inx,len(unadjusted_forecast)):


       unadjusted_forecast[i]=unadjusted_forecast[i-1]+(0.05*unadjusted_forecast[i-1])



    return unadjusted_forecast







def forecastConfirmedCases(request):
    from sklearn.preprocessing import MinMaxScaler
    import pandas as pd
    forecasting_model = load_model('./PredictionModelV4.h5')
    classification_model = load_model('./COV19ClassificationModel.h5')

    weather_occasion = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)),'Egypt_Occasion_Waether_Data.csv'),index_col='Date')

    print("Hi")

    predictions = classification_model.predict(
        x=weather_occasion
        , batch_size=10
        , verbose=0
    )

    rounded_predictions = np.argmax(predictions, axis=-1).tolist()
    weather_occasion['predictions'] = rounded_predictions


    egypt = Uk.objects.all()
    date = []
    confirmed = []
    for row in egypt:
        date.append(row.Date)
        confirmed.append(row.confirmed_cases)
    df_confirmed_country = pd.DataFrame(index=date)
    df_confirmed_country['confirmed'] = confirmed
    df_confirmed_country = df_confirmed_country[:-1]
    train_set = df_confirmed_country.iloc[:math.ceil(98 / 100 * len(df_confirmed_country))]
    test_set = df_confirmed_country.iloc[math.ceil(98 / 100 * len(df_confirmed_country)):]
    date_time_obj = test_set.index.tolist()[0]
    n_input = 45
    n_feature = 1
    scaler = MinMaxScaler()
    scaler.fit(train_set)
    scaled_train = scaler.transform(train_set)
    test_predictions = []
    forecast_date = []
    first_eval_batch = scaled_train[-n_input:]
    current_batch = first_eval_batch.reshape((1, n_input, n_feature))

    for i in range(len(test_set) + 21):
        # get the prediction value for the first batch
        current_pred = forecasting_model.predict(current_batch)[0]
        # append the prediction into the array
        test_predictions.append(current_pred)
        forecast_date.append(str(date_time_obj))
        date_time_obj += timedelta(days=1)
        # use the prediction to update the batch and remove the first value
        current_batch = np.append(current_batch[:, 1:, :], [[current_pred]], axis=1)

    true_predictions = scaler.inverse_transform(test_predictions)



    x = {
        "Date": forecast_date,
        "Forecast": sum(true_predictions.tolist(), []),
        "adjusted": adjustForecast(sum(true_predictions.tolist(), []),weather_occasion)
    }
    time.sleep(10)
    return JsonResponse(x)


def get_egypt_date(request):
    uk = Uk.objects.all()
    qs_json = serializers.serialize('json', uk)
    return HttpResponse(qs_json, content_type='application/json')


def isDatabaseUpdateStatus(date):
    querySet = metaData.objects.all()
    return True if date == querySet[0].Date else querySet[0].Date


def reverseCommulitive(df, column_name):
    observation = df.iloc[:, 0].tolist()
    newReversedCommulitive = [0]
    for i in range(1, len(observation)):
        newReversedCommulitive.append(observation[i] - observation[i - 1])
    dic = {column_name: 'cumulative ' + column_name}
    df.rename(columns=dic, inplace=True)
    df[column_name] = newReversedCommulitive
    return df[1:]


def update_country_data(country, date, df_cofermed, df_deathes, df_recovered):
    d2 = datetime.datetime.strptime(str(date), '%Y-%m-%d').strftime('%#m/%#d/%Y')
    date_range_formated = d2[:-4] + str(int(d2[-4:]) - 2000)
    confermed = reverseCommulitive(df_cofermed[date_range_formated:], 'confirmed')['confirmed'].tolist()
    commultive_confirmed = reverseCommulitive(df_cofermed[date_range_formated:], 'confirmed')[
        'cumulative confirmed'].tolist()
    deathes = reverseCommulitive(df_deathes[date_range_formated:], 'deaths')['deaths'].tolist()
    commultive_deathes = reverseCommulitive(df_deathes[date_range_formated:], 'deaths')[
        'cumulative deaths'].tolist()
    recovered = reverseCommulitive(df_recovered[date_range_formated:], 'recovered')['recovered'].tolist()
    commultive_recovered = reverseCommulitive(df_recovered[date_range_formated:], 'recovered')[
        'cumulative recovered'].tolist()
    for i in range(len(confermed)):
        date += timedelta(days=1)
        cursor = connection.cursor()
        cursor.execute(
            "INSERT INTO  public.\"COV19Predictor_uk\" VALUES  (DEFAULT ,%(date)s ,%(cumulative_confirmed_cases)s,%(confirmed_cases)s,%(cumulative_recovered_cases)s,%(recovered_cases)s,%(cumulative_death_cases)s,%(death_cases)s)",
            {
                'date': date,
                'cumulative_confirmed_cases': commultive_confirmed[i],
                'confirmed_cases': confermed[i],
                'cumulative_recovered_cases': commultive_recovered[i],
                'recovered_cases': recovered[i],
                'cumulative_death_cases': commultive_deathes[i],
                'death_cases': deathes[i]
            });
    return date


def update_lastupadate_meta_data(date):
    metaData.objects.filter().update(Date=date)
    return 1


def checkData(request):
    country = "Egypt"
    df_confirmed = pd.read_csv(
        "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv")
    df_confirmed_country = df_confirmed[df_confirmed["Country/Region"] == country]
    df_confirmed_country = pd.DataFrame(df_confirmed_country[df_confirmed_country.columns[4:]].sum(),
                                        columns=["confirmed"])

    df_deathes = pd.read_csv(
        "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv")
    df_deathes_country = df_deathes[df_deathes["Country/Region"] == country]
    df_deathes_country = pd.DataFrame(df_deathes_country[df_deathes_country.columns[4:]].sum(), columns=["deaths"])

    df_recovred = pd.read_csv(
        "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv")
    df_recovred_country = df_recovred[df_recovred["Country/Region"] == country]
    df_recovred_country = pd.DataFrame(df_recovred_country[df_recovred_country.columns[4:]].sum(),
                                       columns=["recovered"])

    date = isDatabaseUpdateStatus(datetime.strptime(df_confirmed_country.index.tolist()[-1], '%m/%d/%y').date())
    if date != True:
        print("update now")
        new_date = update_country_data("Egypt", date, df_confirmed_country, df_deathes_country, df_recovred_country)
        update_lastupadate_meta_data(new_date)
    else:
        print("data up to date")

    return HttpResponse("Hello")



def create_table(country_name):

    cursor = connection.cursor()
    cursor.execute(
        "create table if not exists public.\""+country_name+
        "\" ( id serial not null primary key ,"
        "\"Date\" date not null,"
        "cumulative_confirmed_cases integer not null,"
        "confirmed_cases integer not null,"
        "cumulative_recovered_cases integer not null,"
        "recovered_cases integer not null,"
        "cumulative_death_cases integer not null,"
        "death_cases integer not null ) ");



def fill_countries_table (country):
    print(country)
    cursor = connection.cursor()
    cursor.execute("INSERT INTO public.country_names values (DEFAULT,%(country)s)",{'country':country})







def create_countries_table(request):
    import pandas as pd
    import numpy as np
    df = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv")


    df_confirmed = pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv")
    df_deathes=pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv")
    df_confirmed=pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv")
    #to be continued
    df = df.replace(np.nan, '', regex=True)
    states = df['Province/State'].tolist()
    countries = df['Country/Region'].tolist()
    print(states[0] == '')

    for i in range(len(states)):
        if states[i] != '':
            count_name = countries[i] + '/' + states[i]
            fill_countries_table(count_name)
            #create_table(count_name)

        else:
            count_name = countries[i] + states[i]
            fill_countries_table(count_name)
            #create_table(count_name)
    return HttpResponse ("Done :D")


def get_countries(request):
    cursor = connection.cursor()
    cursor.execute("select  * from  public.country_names")
    insert_Countries(
        cursor.fetchall(),
        pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"),
        pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"),
        pd.read_csv("https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv")

    )
    return HttpResponse (":wD")


def insert_Countries (countries,df_confirmed,df_deathes,df_recovred):
    for recoreded_country in countries:
        print(">> ",recoreded_country)
        arr = recoreded_country[1].split("/")
        country = arr[0]
        Province_State = ""
        if len(arr) != 1:
            Province_State = arr[1]
            country
        # get country confirmed data
        df_confirmed_country = df_confirmed[df_confirmed["Country/Region"] == country]
        if Province_State != '':
            df_confirmed_country = df_confirmed[df_confirmed["Province/State"] == Province_State]
        df_confirmed_country = pd.DataFrame(df_confirmed_country[df_confirmed_country.columns[4:]].sum(),columns=["confirmed"])
        # get country deaths data

        df_deathes_country = df_deathes[df_deathes["Country/Region"] == country]
        if Province_State != '':
            df_deathes_country = df_deathes[df_deathes["Province/State"] == Province_State]
        df_deathes_country = pd.DataFrame(df_deathes_country[df_deathes_country.columns[4:]].sum(), columns=["deaths"])
        # get country recovered data

        df_recovred_country = df_recovred[df_recovred["Country/Region"] == country]
        if Province_State != '':
            df_recovred_country = df_recovred[df_recovred["Province/State"] == Province_State]
        df_recovred_country = pd.DataFrame(df_recovred_country[df_recovred_country.columns[4:]].sum(),columns=["recovered"])

        df_confirmed_country = reverseCommulitive(df_confirmed_country, 'confirmed')
        df_deathes_country = reverseCommulitive(df_deathes_country, 'deaths')
        df_recovred_country = reverseCommulitive(df_recovred_country, 'recovered')

        country_data = pd.DataFrame()
        dates = df_confirmed_country.index.tolist()
        for i in range(len(dates)):
            dates[i] = datetime.strptime(dates[i], '%m/%d/%y').date()
        country_data['Date'] = dates

        country_data['cumulative confirmed'] = df_confirmed_country['cumulative confirmed'].tolist()
        country_data['confirmed'] = df_confirmed_country['confirmed'].tolist()

        country_data['cumulative deaths'] = df_deathes_country['cumulative deaths'].tolist()
        country_data['deaths'] = df_deathes_country['deaths'].tolist()

        country_data['cumulative recovered'] = df_recovred_country['cumulative recovered'].tolist()
        country_data['recovered'] = df_recovred_country['recovered'].tolist()

        cursor = connection.cursor()
        for index, row in country_data.iterrows():
            cursor.execute(
                "INSERT INTO  public.\""+recoreded_country[1]+ "\"VALUES  (DEFAULT ,%(date)s ,%(cumulative_confirmed_cases)s,%(confirmed_cases)s,%(cumulative_recovered_cases)s,%(recovered_cases)s,%(cumulative_death_cases)s,%(death_cases)s)",
                {
                    'date': row['Date'],
                    'cumulative_confirmed_cases':row['cumulative confirmed'] ,
                    'confirmed_cases': row['confirmed'],
                    'cumulative_recovered_cases': row['cumulative recovered'],

                    'recovered_cases': row['recovered'],
                    'cumulative_death_cases': row['cumulative deaths'],
                    'death_cases': row['deaths']
                });



        time.sleep(0.75)
