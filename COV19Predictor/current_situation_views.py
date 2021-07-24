import json

from django.db import connection
from django.http import HttpResponse, JsonResponse
from django.shortcuts import render


def get_countries_list(request):
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM public.\"country_names\"")
    result_set = cursor.fetchall()
    countries = []
    for row in result_set:
        countries.append(row[1])
    return HttpResponse(json.dumps(countries), content_type='application/json')




