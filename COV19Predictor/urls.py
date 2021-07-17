

from django.urls import path

from . import views

urlpatterns = [
    path('', views.getPrediction),
    path('getActualConfirmed/',views.getActualConfirmed),
    path('forecastConfirmedCases/',views.forecastConfirmedCases),
    path('getActualDeath/',views.getActualDeath),
    path('getActualRecovered/', views.getActualRecovered)

]
