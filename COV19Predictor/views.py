import datetime
import os
import time
from django.core import serializers
from django.db import connection
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
import math
from datetime import timedelta, datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from .models import Uk
from .models import metaData
import itertools as it


def getPrediction(request):
    """
    function to initialize forecasting template
    @rtype: object
    """
    return render(request, 'Hello.html');


def find_ranges(lst, n=4, number=1):
    """Return ranges for `n` or more repeated values."""
    groups = ((k, tuple(g)) for k, g in it.groupby(enumerate(lst), lambda x: x[-1]))
    repeated = (idx_g for k, idx_g in groups if len(idx_g) >= n)
    rebeats = ((sub[0][0], sub[-1][0]) for sub in repeated)
    one_rebeats = []
    for r in rebeats:
        if lst[r[0]] == 1:
            one_rebeats.append(r)
    return one_rebeats


def adjustForecast(unadjusted_forecast, adjusted_helper):
    """
    function to perform forecast adjustment
    @rtype: object
    """
    start_inx = list(find_ranges(adjusted_helper['predictions'], 5))[0][0]
    for i in range(start_inx, len(unadjusted_forecast)):
        unadjusted_forecast[i] = unadjusted_forecast[i - 1] + (0.05 * unadjusted_forecast[i - 1])
    return unadjusted_forecast


def forecastConfirmedCases(request):
    """
    function to perform basic forecasting for Egypt
    @rtype: object
    """
    forecasting_model = load_model('./PredictionModelV4.h5')
    classification_model = load_model('./COV19ClassificationModel.h5')
    weather_occasion = pd.read_csv(
        os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Egypt_Occasion_Waether_Data.csv'), index_col='Date')
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
        "adjusted": adjustForecast(sum(true_predictions.tolist(), []), weather_occasion)
    }

    return JsonResponse(x)


def get_egypt_date(request):
    uk = Uk.objects.all()
    qs_json = serializers.serialize('json', uk)
    return HttpResponse(qs_json, content_type='application/json')


def isDatabaseUpdateStatus(date):
    """
    function to check whether the database is up to date or not
    @param date:
    @return:
    """
    querySet = metaData.objects.all()
    return True if date == querySet[0].Date else querySet[0].Date


def reverseCumulative(df, column_name):
    """
    function to reverse cumulative statistic for dataframe
    @param df:
    @param column_name:
    @return:
    """
    observation = df.iloc[:, 0].tolist()
    reverse_cumulative = [0]
    for i in range(1, len(observation)):
        reverse_cumulative.append(observation[i] - observation[i - 1])
    dic = {column_name: 'cumulative' + ' ' + column_name}
    df.rename(columns=dic, inplace=True)
    df[column_name] = reverse_cumulative
    return df[1:]


def update_last_update_meta_data(date):
    """
    function to update last update time for countries DB
    @rtype: object
    """
    metaData.objects.filter().update(Date=date)
    return 1


def create_table(country_name):
    """
    function to execute the actual SQL create statement for country table creation
    @param:country_name
    @rtype: object
    """
    cursor = connection.cursor()
    cursor.execute(
        "create table if not exists public.\"" + country_name +
        "\" ( id serial not null primary key ,"
        "\"Date\" date not null,"
        "cumulative_confirmed_cases integer not null,"
        "confirmed_cases integer not null,"
        "cumulative_recovered_cases integer not null,"
        "recovered_cases integer not null,"
        "cumulative_death_cases integer not null,"
        "death_cases integer not null ) "
    );


def get_last_dataframe_date():
    """
    function to return the last date in the county data frame > to compare it with last_update date
    @rtype: object
    """
    df_confirmed = pd.read_csv(
        "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv")
    df_confirmed_country = df_confirmed[df_confirmed["Country/Region"] == "Egypt"]
    df_confirmed_country = pd.DataFrame(df_confirmed_country[df_confirmed_country.columns[4:]].sum(),
                                        columns=["confirmed"])
    return df_confirmed_country.index.tolist()[-1]


def get_countries(request):
    date = isDatabaseUpdateStatus(datetime.strptime(get_last_dataframe_date(), '%m/%d/%y').date())
    if date != True:
        print("update now")
        cursor = connection.cursor()
        cursor.execute("select  * from  public.country_names")
        insert_Countries(
            cursor.fetchall(),
            pd.read_csv(
                "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"),
            pd.read_csv(
                "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"),
            pd.read_csv(
                "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv")
            , date, datetime.strptime(get_last_dataframe_date(), '%m/%d/%y').date()
        )

    else:
        print("data up to date")

    return HttpResponse("")


def insert_Countries(countries, df_confirmed, df_deaths, df_recovered, start_date, last_date, initialize=None):
    """
    function to populate countries statistics on its corresponding table
    @param countries: all system available countries
    @param df_confirmed:
    @param df_deaths:
    @param df_recovered:
    @param start_date:
    @param last_date:
    @param initialize:
    """
    for recoreded_country in countries:
        print(">> ", recoreded_country)
        arr = recoreded_country[1].split("/")
        country = arr[0]
        Province_State = ""
        if len(arr) != 1:
            Province_State = arr[1]

        # get country confirmed data
        df_confirmed_country = df_confirmed[df_confirmed["Country/Region"] == country]
        if Province_State != '':
            df_confirmed_country = df_confirmed[df_confirmed["Province/State"] == Province_State]
        df_confirmed_country = pd.DataFrame(df_confirmed_country[df_confirmed_country.columns[4:]].sum(),
                                            columns=["confirmed"])

        # get country deaths data
        df_deathes_country = df_deaths[df_deaths["Country/Region"] == country]
        if Province_State != '':
            df_deathes_country = df_deaths[df_deaths["Province/State"] == Province_State]
        df_deathes_country = pd.DataFrame(df_deathes_country[df_deathes_country.columns[4:]].sum(), columns=["deaths"])

        # get country recovered data
        df_recovred_country = df_recovered[df_recovered["Country/Region"] == country]
        if Province_State != '':
            df_recovred_country = df_recovered[df_recovered["Province/State"] == Province_State]
        df_recovred_country = pd.DataFrame(df_recovred_country[df_recovred_country.columns[4:]].sum(),
                                           columns=["recovered"])

        # reverse commulitives and add it
        df_confirmed_country = reverseCumulative(df_confirmed_country, 'confirmed')
        df_deathes_country = reverseCumulative(df_deathes_country, 'deaths')
        df_recovred_country = reverseCumulative(df_recovred_country, 'recovered')

        country_data = pd.DataFrame()

        # format date column
        dates = df_confirmed_country.index.tolist()
        for i in range(len(dates)):
            dates[i] = datetime.strptime(dates[i], '%m/%d/%y').date()

        # construct the complete dataframe for country
        country_data.index = country_data['Date']
        country_data['Date'] = dates
        country_data['cumulative confirmed'] = df_confirmed_country['cumulative confirmed'].tolist()
        country_data['confirmed'] = df_confirmed_country['confirmed'].tolist()
        country_data['cumulative deaths'] = df_deathes_country['cumulative deaths'].tolist()
        country_data['deaths'] = df_deathes_country['deaths'].tolist()
        country_data['cumulative recovered'] = df_recovred_country['cumulative recovered'].tolist()
        country_data['recovered'] = df_recovred_country['recovered'].tolist()

        start_date += timedelta(days=1)
        # get the specified time frame rom dataframe
        country_data = country_data[start_date:]

        cursor = connection.cursor()
        for index, row in country_data.iterrows():
            cursor.execute(
                "INSERT INTO  public.\"" + recoreded_country[1] + "\"VALUES  (DEFAULT ,%(date)s ,%(cumulative_confirmed_cases)s,%(confirmed_cases)s,%(cumulative_recovered_cases)s,%(recovered_cases)s,%(cumulative_death_cases)s,%(death_cases)s)",
                {
                    'date': row['Date'],
                    'cumulative_confirmed_cases': row['cumulative confirmed'],
                    'confirmed_cases': row['confirmed'],
                    'cumulative_recovered_cases': row['cumulative recovered'],
                    'recovered_cases': row['recovered'],
                    'cumulative_death_cases': row['cumulative deaths'],
                    'death_cases': row['deaths']
                });

    update_last_update_meta_data(last_date)



# dangerous method used for development purpose
def emptyCountriesTable(request):
    """
    function to empty all countries table, it's so dangerous but used for debugging purpose
    @param request:
    @return:
    """
    cursor = connection.cursor()
    cursor.execute("select  * from  public.country_names")
    results = cursor.fetchall()
    for result in results:
        print(result[1])
        cursor.execute("DELETE  FROM  public.\"" + result[1] + "\" ;")
    return HttpResponse("All countries table are initialized successfully")


# dangerous method used for development purpose
def create_countries_table(request):
    """

    function to create and initialize DB table for all countries
    uncomment create_table() method to create table
    uncomment fill_countries_table() to add the newly added country to list of viable countries
    @note this method is used one time to create db table
    @param request:
    @return:
    """
    import pandas as pd
    import numpy as np
    df = pd.read_csv(
        "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv")
    df = df.replace(np.nan, '', regex=True)
    states = df['Province/State'].tolist()
    countries = df['Country/Region'].tolist()
    for i in range(len(states)):
        if states[i] != '':
            country_name = countries[i] + '/' + states[i]
            create_table(country_name)
            # fill_countries_table(count_name)
        else:
            country_name = countries[i] + states[i]
            create_table(country_name)
            # fill_countries_table(count_name)
    return HttpResponse("countries tables has been created successfully")
