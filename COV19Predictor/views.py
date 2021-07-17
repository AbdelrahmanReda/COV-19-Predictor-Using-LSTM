import math
from datetime import timedelta

import joblib
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
import pandas as pd
import json
import numpy as np

# Create your views here.
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.python.keras.preprocessing.sequence import TimeseriesGenerator


def getPrediction(request):
    return render(request, 'Hello.html');


country = "Egypt"
df_confirmed = pd.read_csv(
    "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv")
# df_confirmed.to_csv('global.csv')
df_confirmed_country = df_confirmed[df_confirmed["Country/Region"] == country]
df_confirmed_country = pd.DataFrame(df_confirmed_country[df_confirmed_country.columns[4:]].sum(), columns=["confirmed"])
df_confirmed_country.index = pd.to_datetime(df_confirmed_country.index, format='%m/%d/%y')


def getActualConfirmed(request):
    data = []
    for index, row in df_confirmed_country.iterrows():
        data.append([index, row['confirmed']])

    confirmedCases = []
    date = []
    for i in range(1, len(data)):
        confirmedCases.append(int(data[i][1] - data[i - 1][1]))
        date.append(data[i][0].strftime('%Y/%m/%d'))
    x = {
        'date': date,
        'confirmed': confirmedCases

    }
    return JsonResponse(x)


def forecastConfirmedCases(request):
    model = load_model('./predictionModelV2.h5')
    print(model.summary())

    country = "Egypt"

    # Total COVID confirmed cases
    df_confirmed = pd.read_csv(
        "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv")
    df_confirmed_country = df_confirmed[df_confirmed["Country/Region"] == country]
    df_confirmed_country = pd.DataFrame(df_confirmed_country[df_confirmed_country.columns[4:]].sum(),
                                        columns=["confirmed"])
    df_confirmed_country['res'] = df_confirmed_country['confirmed'].diff().fillna(
        df_confirmed_country['confirmed']).astype(int)
    del df_confirmed_country['confirmed']
    df_confirmed_country.rename({"res": "confirmed"}, axis='columns', inplace=True)

    df = df_confirmed_country
    df.index = pd.to_datetime(df.index)

    # training and testing
    train = df.iloc[:math.floor(80 / 100 * len(df))]
    test = df.iloc[math.floor(80 / 100 * len(df)):]

    scaler = MinMaxScaler()
    scaler.fit(train)
    scaled_train = scaler.transform(train)
    scaler.transform(test)

    # generator dif
    n_input = 15
    n_features = 1

    import tensorflow as tf;

    model = tf.keras.models.load_model('predictionModelV2.h5')

    last_train_batch = scaled_train[-15:]
    last_train_batch = last_train_batch.reshape((1, n_input, n_features))
    model.predict(last_train_batch)

    test_prediction = []
    first_eval_batch = scaled_train[-n_input:]
    current_batch = first_eval_batch.reshape((1, n_input, n_features))
    for i in range(len(test) + 15):
        current_prediction = model.predict(current_batch)[0]
        test_prediction.append(current_prediction)
        current_batch = np.append(current_batch[:, 1:, :], [[current_prediction]], axis=1)
    reversed_scaled_predicitons = scaler.inverse_transform(test_prediction)

    date = []
    startData = test.index.tolist()[0]

    for i in range(len(test) + 15):
        date.append(startData)
        startData += timedelta(days=1)

    print(date)
    print(reversed_scaled_predicitons)
    forecast = []
    for element in reversed_scaled_predicitons:
        forecast.append(math.floor(element))

    forecast_date = {
        "date": date,
        "forecast-confirmed": forecast

    }

    return JsonResponse(forecast_date)


def getActualDeath(request):
    country = "Egypt"
    df_confirmed = pd.read_csv(
        "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv")
    # df_confirmed.to_csv('global.csv')
    df_confirmed_country = df_confirmed[df_confirmed["Country/Region"] == country]
    df_confirmed_country = pd.DataFrame(df_confirmed_country[df_confirmed_country.columns[4:]].sum(),
                                        columns=["death"])
    data = []
    for index, row in df_confirmed_country.iterrows():
        data.append([index, row['death']])
    death = []
    date = []
    for i in range(1, len(data)):
        death.append(int(data[i][1] - data[i - 1][1]))
        date.append(data[i][0])
    x = {
        'death': death,
        'date': date,
    }
    return JsonResponse(x);


def getActualRecovered(request):
    country = "Egypt"
    df_confirmed = pd.read_csv(
        "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv")
    # df_confirmed.to_csv('global.csv')
    df_confirmed_country = df_confirmed[df_confirmed["Country/Region"] == country]
    df_confirmed_country = pd.DataFrame(df_confirmed_country[df_confirmed_country.columns[4:]].sum(),
                                        columns=["recovered"])
    data = []
    for index, row in df_confirmed_country.iterrows():
        data.append([index, row['recovered']])
    death = []
    date = []
    for i in range(1, len(data)):
        death.append(int(data[i][1] - data[i - 1][1]))
        date.append(data[i][0])
    x = {
        'recovered': death,
        'date': date,
    }
    return JsonResponse(x);


def returnForecastedDeath(request):
    return 0;
