import folium
import os
from shapely.wkt import loads
import urllib.parse
import pickle


from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.contrib.gis.geos import GEOSGeometry, Point
from django.contrib import messages
from django.http import HttpResponseRedirect, HttpRequest, HttpResponse
from django.urls import reverse


from rails.models import Station, Road
from rails.utils.files_parse import process_xls_file
from rails.m_learning.modeling import create_models, update_models


def road_area_detail(request: HttpRequest, area_id: int) -> HttpResponse:
    def marker_color(geometry: GEOSGeometry, point: Point) -> str:
        if geometry.contains(point):
            return 'green'
        else:
            return 'red'
    try:
        road = Road.objects.get(id=area_id)
    except Road.DoesNotExist:
        return render(request, '404.html', {'message': 'Road not found'})

    road_area = road.area

    shapely_geometry = loads(road_area.wkt)

    m = folium.Map(location=[61, 105], zoom_start=4)
    folium.GeoJson(shapely_geometry).add_to(m)

    for station in Station.objects.filter(road=road):
        folium.Marker(location=station.coords, popup=str(station.id) + ' ' + station.name,
                      icon=folium.Icon(color=marker_color(road_area, station.point))).add_to(m)

    map_html = m._repr_html_()

    return render(request, 'road_map.html', {'map_html': map_html})


def process_files(request: HttpRequest) -> HttpResponse:
    problem_stations = []
    files = request.GET.get('files', '').split(',')

    temp_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'temp_xls')
    fs = FileSystemStorage(location=temp_folder)
    for filename in files:
        correct_filename = urllib.parse.unquote(filename)
        source = None
        if 'Объедин'.upper() in correct_filename.upper():
            source = 'rzhd'
            # problem_stations += process_xls_file(correct_filename)
        elif 'для ИНК ' in correct_filename or 'ИНК 4' in correct_filename:
            source = 'tts'
        elif 'Дислокация_' in correct_filename:
            source = 'komtrans'
        elif 'Дислокация вагонов' in correct_filename:
            source = 'sgtrans'
        if source:
            problem_stations += process_xls_file(correct_filename, source)
        fs.delete(correct_filename)
        messages.info(request, 'file processed ' + correct_filename)
    return render(request, 'processed.html', {'problem_stations': problem_stations})


def upload_xls_file(request: HttpRequest) -> HttpResponse:
    def get_filename(uploaded_file):
        if isinstance(uploaded_file.name, bytes):
            return uploaded_file.name.decode('utf-8')
        else:
            return uploaded_file.name

    messages.info(request, 'Загрузка файлов')
    if request.method == 'POST' and 'files' in request.FILES:
        uploaded_files = request.FILES.getlist('files')
        files = []
        if uploaded_files:
            messages.info(request, 'files selected')
            fs = FileSystemStorage()
            temp_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'temp_xls')
            if not os.path.exists(temp_folder):
                os.makedirs(temp_folder)
            for uploaded_file in uploaded_files:
                uploaded_filename = get_filename(uploaded_file)
                filename = urllib.parse.quote(uploaded_filename)
                print(filename)
                file_path = os.path.join(temp_folder, filename)
                uploaded_file_path = os.path.join(temp_folder, uploaded_filename)
                files += [file_path]
                fs.save(uploaded_file_path, uploaded_file)

        url = reverse('process_files') + '?files=' + ','.join(files)
        print('url ', url)
        return HttpResponseRedirect(url)

    return render(request, 'upload_xls.html')


def create_rails_models(request: HttpRequest) -> HttpResponse:
    with open('routes.pkl', 'rb') as f:
        routes = pickle.load(f)
    models = create_models(routes)

    return HttpResponse('models created')


def update_rails_models(request: HttpRequest) -> HttpResponse:
    with open('routes.pkl', 'rb') as f:
        routes = pickle.load(f)
    models = update_models(routes)

    return HttpResponse('models updated')
