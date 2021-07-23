

from django.urls import path

from . import views

urlpatterns = [
    path('', views.getPrediction),

    path('forecastConfirmedCases/',views.forecastConfirmedCases),




    path('updater/', views.update_last_update_meta_data),
    path('getegypt/',views.get_egypt_date),
    path('create_countries_table',views.create_countries_table),
    path('getAllCountries',views.get_countries),
    path('emptContriesTable', views.emptyCountriesTable)






]
