import pandas as pd
from datetime import datetime
from asgiref.sync import async_to_sync

from channels.layers import get_channel_layer

from rails.models import Station, Route
from rails.consumers import ViewStatusConsumer


def arrival_date(locations: dict[str, dict[str, str]]) -> str | None:
    selected_keys = [key for key, value in locations.items() if value['ops_station_id'] == 1071]
    if selected_keys == []:
        return None
    return min(selected_keys)


def time_to_home(update: datetime | str, arrival: datetime | str) -> int:
    date1 = datetime.strptime(update, '%Y-%m-%d')
    date2 = datetime.strptime(arrival, '%Y-%m-%d')
    difference = date2 - date1
    return difference.days


def get_data_from_bd() -> pd.DataFrame:
    home_station = Station.objects.get(id=1071)
    home_routes = Route.objects.filter(end=home_station).exclude(start=home_station)

    routes = pd.DataFrame(columns=['num', 'update', 'route', 'route_id', 'station', 'station_id',
                                   'lat', 'lon', 'd_left', 'time_to_home', 'start_date', 'arrival'])

    total = len(home_routes)
    consumer = ViewStatusConsumer()
    channel_layer = get_channel_layer()
    async_to_sync(channel_layer.group_add)('view_status', consumer.channel_name)

    for idx, route in enumerate(home_routes):
        message = f" processing {idx+1} of {total} - route {route}"
        async_to_sync(channel_layer.group_send)('view_status', {'type': 'view_status_update', 'message': message})

        for update, values in route.locations.items():

            num = route.wagon.num
            station = Station.objects.get(id=values['ops_station_id'])
            if station == home_station:
                continue
            lat, lon = station.coords
            arrival = arrival_date(route.locations)
            if arrival is None:
                continue
            d_left = station.distances_left.get(str(home_station.id), None)
            if d_left is None:
                continue
            to_home = time_to_home(update, arrival)
            if to_home < 0:
                continue
            start_date = route.start_date
            data = {'num': num, 'update': update, 'lat': lat, 'lon': lon, 'd_left': d_left,
                    'time_to_home': to_home, 'start_date': start_date, 'arrival': arrival,
                    'route': str(route), 'station': station.name, 'route_id': route.id, 'station_id': station.id}
            index = num + '_' + update
            routes.loc[index] = data
    return routes
