import pandas as pd
import re
import datetime
from typing import Any
import requests

from rails.models import Station, Road, Wagon, Cargo


def get_code(location: str) -> str | None:
    pattern = r'\((\d+)\)[^(]*$'
    if not isinstance(location, str):
        location = str(location)
    match = re.search(pattern, location)
    if match:
        return match.group(1)
    else:
        return None


def get_date(date: Any) -> str | None:

    def excel_to_datetime(excel_float):
        return datetime.datetime(1899, 12, 30) + datetime.timedelta(days=excel_float)

    if date == 'NaT' or date == '' or date is None or pd.isnull(date) or pd.isna(date) or date == 'NaT':
        return None

    if isinstance(date, (float, int)):
        return excel_to_datetime(date).strftime('%Y-%m-%d')

    elif isinstance(date, datetime.datetime):
        return date.strftime('%Y-%m-%d')

    elif isinstance(date, pd.Timestamp):
        date = datetime.datetime.strptime(date, "%d.%m.%Y %H:%M")
        return date.strftime('%Y-%m-%d')
    else:
        try:
            return datetime.datetime.strptime(str(date), '%d.%m.%Y %H:%M').strftime('%Y-%m-%d')
        except ValueError:
            pass
        try:
            return datetime.datetime.strptime(str(date), '%d.%m.%Y %H:%M:%S').strftime('%Y-%m-%d')
        except ValueError:

            return datetime.datetime.strptime(str(date), '%d.%m.%Y').strftime('%Y-%m-%d')


def get_num(wagon_num: Any) -> str | None:
    if wagon_num == 'NaT' or wagon_num == '' or pd.isnull(wagon_num) or pd.isna(wagon_num):
        return None
    if isinstance(wagon_num, str):
        return wagon_num
    elif isinstance(wagon_num, int):
        return str(wagon_num)
    elif isinstance(wagon_num, float):
        return str(int(wagon_num))
    return None


def get_cargo_name(cargo_name: str) -> str | None:
    if cargo_name == 'Пропан (226125)':
        cargo_name = Cargo.PT
    elif cargo_name == 'Бутан или бутана смеси (226040)':
        cargo_name = Cargo.BT
    elif cargo_name == 'Газы углеводородные сжиженные, не поименованные в алфавите (226074)':
        cargo_name = Cargo.PBT
    else:
        cargo_name = None


def fetch_coordinates(station: str) -> tuple[float, float] | None:
    url = 'https://nominatim.openstreetmap.org/search?'
    pattern = r'\([^)]*\)'

    if not isinstance(station, str):
        station = str(station)

    location = re.sub(pattern, "", station).strip()
    params = {'q': location, 'format': 'json', 'railway': 'station, stop, halt'}
    response = requests.get(f"{url}{'&'.join([f'{k}={v}' for k, v in params.items()])}")
    results = response.json()

    try:
        coords = [results[0]['lat'], results[0]['lon']]
        if coords:
            return coords
        else:
            return None
    except Exception:
        return None


def get_wagon(wagon_num: str) -> Wagon:
    try:
        return Wagon.objects.get(num=wagon_num)
    except Wagon.DoesNotExist:
        wagon = Wagon(num=wagon_num)
        wagon.save()
        return wagon


def get_station(station_name: str, item_road: str | None, source: str = 'rzhd', mapping: dict = {}) -> Station | None:
    if not isinstance(station_name, str):
        return None
    code = get_code(station_name)
    rzhd = source == 'rzhd'
    if rzhd and code is None:
        return None
    elif rzhd:
        try:
            return Station.objects.get(code=code)
        except Station.DoesNotExist:
            try:
                road = Road.objects.get(name=item_road)
            except Road.DoesNotExist:
                return None

            station = Station(name=station_name, code=code, road=road)
            coords = fetch_coordinates(station_name)

            if coords is None:
                return None
            else:
                station.coords_lat, station.coords_lon = coords
                station.save()
                return station
    else:
        if source == 'sgtrans':
            station_name = station_name[:-3].rstrip()
        if mapping:
            station_name = mapping.get(station_name.lower(), station_name)
        station_name += ' '
        try:
            return Station.objects.get(name__icontains=station_name.upper())
        except Station.DoesNotExist:
            return None
        except Station.MultipleObjectsReturned:
            stations = Station.objects.filter(name__icontains=station_name.upper()).order_by('name')
            return stations[0]


def extract_date_from_filename(filename: str, source: str) -> str | None:
    if source == 'rzhd':
        match = re.search(r'\d{8}', filename)
        if match:
            date_part = match.group(0)
            return f"{date_part[4:]}-{date_part[2:4]}-{date_part[:2]}"
        else:
            return None
    else:
        match = re.search(r'(\d{2})\.(\d{2})(?:\.(\d{2}))?', filename)
        if match:
            day = match.group(1)
            month = match.group(2)
            year = match.group(3) if match.group(3) else "23"
            year = "20" + year
            return f"{year}-{month}-{day}"
        return None
