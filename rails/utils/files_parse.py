import os

import pandas as pd

from rails.models import Route
from rails.utils.parsing_utils import get_station, get_wagon, get_cargo_name, get_num, get_date, extract_date_from_filename


def process_xls_file(filename: str, source: str) -> list[str] | None:
    columns_of_interest = {
                        'rzhd': ['Номер вагона', 'Дата и время начала рейса', 'Станция отправления', 'Дата и время окончания рейса',
                                 'Станция назначения', 'Вес груза (кг)', 'Станция операции',
                                 'Операция с вагоном', 'Дата и время операции', 'Расстояние оставшееся (км)',
                                 'Дорога отправления', 'Дорога назначения', 'Дорога операции', 'Наименование груза'],
                        'tts': ['Номер вагона', 'Дата нач. рейса', 'Ст. нач. рейса', 'Ст. назн.', 'Станция опер.',
                                'Наименование операции', 'Дата/время опер.', 'Ост. расстояние'],
                        'sgtrans': ['Номер вагона', 'Дата нач. рейса', 'Ст. нач. рейса', 'Ст. назн.', 'Станция опер.',
                                    'Наименование операции', 'Дата опер.', 'Ост. расстояние'],
                        'komtrans': ['Номер вагона', 'Вр. нач. рейса', 'Ст. нач. рейса', 'Ст. назн.', 'Станция опер.',
                                     'Наименование операции', 'Дата/время опер.', 'Ост. расстояние']
                        }

    columns_mapping = {
                    'rzhd': {'Номер вагона': 'wagon_num', 'Дата и время начала рейса': 'route_start',
                             'Станция отправления': 'start_location', 'Дорога отправления': 'start_road',
                             'Дата и время окончания рейса': 'route_end', 'Станция назначения': 'destination_location',
                             'Дорога назначения': 'destination_road', 'Вес груза (кг)': 'weight',
                             'Станция операции': 'ops_station', 'Дорога операции': 'ops_road',
                             'Операция с вагоном': 'ops', 'Дата и время операции': 'ops_date',
                             'Расстояние оставшееся (км)': 'd_left', 'Наименование груза': 'cargo_name'},
                    'tts': {'Номер вагона': 'wagon_num', 'Дата нач. рейса': 'route_start',
                            'Ст. нач. рейса': 'start_location', 'Ст. назн.': 'destination_location',
                            'Станция опер.': 'ops_station', 'Наименование операции': 'ops',
                            'Дата/время опер.': 'ops_date', 'Ост. расстояние': 'd_left'},
                    'sgtrans': {'Номер вагона': 'wagon_num', 'Дата нач. рейса': 'route_start',
                                'Ст. нач. рейса': 'start_location', 'Ст. назн.': 'destination_location',
                                'Станция опер.': 'ops_station', 'Наименование операции': 'ops',
                                'Дата опер.': 'ops_date', 'Ост. расстояние': 'd_left'},
                    'komtrans':    {'Номер вагона': 'wagon_num', 'Вр. нач. рейса': 'route_start',
                                    'Ст. нач. рейса': 'start_location', 'Ст. назн.': 'destination_location',
                                    'Станция опер.': 'ops_station', 'Наименование операции': 'ops',
                                    'Дата/время опер.': 'ops_date', 'Ост. расстояние': 'd_left'}
                            }

    if source not in ['rzhd', 'tts', 'sgtrans', 'komtrans']:
        raise ValueError(f"Source {source} is not supported")
    RUNNING_IN_DOCKER = os.getenv('RUNNING_IN_DOCKER', 'false') == 'true'

    if RUNNING_IN_DOCKER:
        excel_file_path = '/app/rails/utils/stations_mapping.xlsx'
    else:
        excel_file_path = 'rails/utils/stations_mapping.xlsx'

    mapping = pd.read_excel(excel_file_path,
                            header=1).set_index('Значение')['замена'].to_dict()
    if filename.endswith('.xlsx'):
        formated_date = extract_date_from_filename(filename, source)
        if source == 'rzhd':
            header_row = 3
            df = pd.read_excel(filename, header=header_row, usecols=columns_of_interest[source], engine='openpyxl')
        elif source == 'tts':
            excel_file = pd.ExcelFile(filename, engine='openpyxl')
            sheet_names = excel_file.sheet_names
            pattern_1 = "массив"
            pattern_2 = "общ"
            for sheet_name in sheet_names:
                if pattern_1 in sheet_name or pattern_2 in sheet_name:
                    df = excel_file.parse(sheet_name=sheet_name, cols=columns_of_interest[source], engine='openpyxl')
                    break
        elif source == 'sgtrans':
            df = pd.read_excel(filename, usecols=columns_of_interest[source], header=1, engine='openpyxl')
        else:
            df = pd.read_excel(filename, usecols=columns_of_interest[source], engine='openpyxl')
    else:
        return None
    route_df = pd.DataFrame()
    for column in columns_of_interest[source]:
        if column in df.columns:
            route_df[columns_mapping[source][column]] = df[column]
        else:
            route_df[columns_mapping[source][column]] = None

    route_df['index_col'] = route_df['wagon_num'].astype(str) + '_' + str(formated_date)
    route_df.set_index('index_col', inplace=True)

    home_station = get_station('Лена-Восточная', None, source, mapping=mapping)

    problem_stations = []
    update = formated_date

    for _, row in route_df.iterrows():
        wagon_num = get_num(row['wagon_num'])
        if wagon_num is None:
            continue
        wagon = get_wagon(wagon_num)

        ops = row.get('ops')

        ops_road = row.get('ops_road', None)

        ops_date = get_date(row.get('ops_date', None))
        if ops_date is None:
            continue

        ops_station = row.get('ops_station', None)

        destination_location = row.get('destination_location')
        destination_road = row.get('destination_road')

        d_left = row.get('d_left')

        start_location = row.get('start_location', None)
        start_road = row.get('start_road', None)

        route_start = get_date(row.get('route_start', None))

        if route_start is None:
            continue

        route_end = get_date(row.get('route_end', None))

        weight = row.get('weight', None)
        if weight == 'NaT' or weight == '' or pd.isnull(weight) or pd.isna(weight):
            weight = None

        cargo_name = get_cargo_name(row.get('cargo_name', None))
        rzhd = source == 'rzhd'
        start_point = get_station(start_location, start_road, mapping=mapping, source=source)
        dest_point = get_station(destination_location, destination_road, mapping=mapping, source=source)
        ops_point = get_station(ops_station, ops_road, mapping=mapping, source=source)

        if start_point is None:
            if not isinstance(start_location, str):
                start_location = str(start_location)
            problem_stations += ['start ' + start_location]
            continue
        if dest_point is None:
            if not isinstance(destination_location, str):
                destination_location = str(destination_location)
            problem_stations += ['dest ' + destination_location]
            continue
        if ops_point is None:
            if not isinstance(ops_station, str):
                ops_station = str(ops_station)
            problem_stations += ['ops ' + ops_station]
            continue

        check_d_left = ops_point.get_distance_left_for_destination(dest_point.id)

        if check_d_left is None and d_left is not None and rzhd:
            ops_point.set_distance_left_for_destination(dest_point.id, d_left)
            ops_point.save()

        if cargo_name is not None and weight is not None:
            loaded = True
        else:
            loaded = False

        try:
            the_route = Route.objects.get(wagon=wagon, start_date=route_start)
        except Route.DoesNotExist:
            if rzhd or start_point == home_station or dest_point == home_station:
                the_route = Route(
                    wagon=wagon, cargo=cargo_name, weight=weight,
                    start=start_point, end=dest_point, start_date=route_start, end_date=route_end,
                    loaded=loaded)
                the_route.locations = {}
                the_route.save()
            else:
                the_route = None
            problem_stations += ['no route ' + wagon_num + ' ' + str(route_start) + str(start_point) + ' ' + str(dest_point)]

        try:
            if the_route is not None:
                the_route.set_ops_for_date(update, ops_point, ops_date, ops)
                the_route.save()
        except ValueError:
            pass
    return problem_stations


if __name__ == '__main__':
    pass
"""
    folder_path = '/Users/sergeykuzmin/projects/rails/rails/wagons_data/rzhd'
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.xlsx'):
            print(file_name)
            filename = os.path.join(folder_path, file_name)
            problem_stations = process_xls_file(filename)
    komtrans = '/Users/sergeykuzmin/projects/rails/rails/wagons_data/komtrans'
    tts = '/Users/sergeykuzmin/projects/rails/rails/wagons_data/tts'
    sgtrans = '/Users/sergeykuzmin/projects/rails/rails/wagons_data/sgtrans'
    folder_path = tts
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.xlsx'):
            print(file_name)
            filename = os.path.join(folder_path, file_name)
            problem_stations = process_other_file(None, filename, source='tts')
    folder_path = sgtrans
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.xlsx'):
            print(file_name)
            filename = os.path.join(folder_path, file_name)
            problem_stations = process_other_file(None, filename, source='sgtrans')

    folder_path = komtrans
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.xlsx'):
            print(file_name)
            filename = os.path.join(folder_path, file_name)
            problem_stations = process_other_file(None, filename, source='komtrans')
"""
