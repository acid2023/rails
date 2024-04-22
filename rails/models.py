from enum import Enum

from django.db import models
from django.contrib.gis.db.models import GeometryField
from django.contrib.gis.geos import Point


class Road(models.Model):
    name = models.CharField(max_length=512)
    area = GeometryField(null=True, blank=True)

    def is_valid(self, point: tuple[float, float] | Point) -> bool:
        def get_float(value: str | float) -> float:
            if isinstance(value, str):
                return float(value)
            return value

        if not isinstance(point, Point):
            lat = get_float(point[0])
            lon = get_float(point[1])
            geo_point = Point(lon, lat)
        else:
            geo_point = point

        return self.area.contains(geo_point)


class Station(models.Model):
    name = models.CharField(max_length=512)
    code = models.CharField(max_length=8)
    road = models.ForeignKey('Road', on_delete=models.PROTECT, null=True, blank=True)
    coords_lat = models.FloatField(null=True, blank=True)
    coords_lon = models.FloatField(null=True, blank=True)
    distances_left = models.JSONField(null=True, blank=True)

    def get_distance_left_for_destination(self, destination_station_id: int) -> float | int | None:
        if self.distances_left is None:
            self.distances_left = {}
            return None
        return self.distances_left.get(str(destination_station_id), None)

    def set_distance_left_for_destination(self, destination_station_id: int, distance_left: int | float) -> None:
        if self.distances_left is None:
            self.distances_left = {}
        self.distances_left[str(destination_station_id)] = distance_left

    def __str__(self) -> str:
        return f"{self.name} ({self.code}): {self.coords_lat}, {self.coords_lon}"

    @property
    def coords(self) -> tuple[float, float]:
        return (self.coords_lat, self.coords_lon)

    @property
    def point(self) -> Point:
        return Point(self.coords_lon, self.coords_lat)


class Wagon(models.Model):
    num = models.CharField(max_length=8)
    owner = models.CharField(max_length=512, blank=True, null=True)


class Cargo(Enum):
    BT = 'butan'
    PT = 'propan'
    PBT = 'propan-butan'
    BT_45 = 'butan-45'
    BT_75 = 'butan-75'


class Route(models.Model):
    wagon = models.ForeignKey('Wagon', on_delete=models.PROTECT)
    cargo_choices = [(cargo.value, cargo.name) for cargo in Cargo]
    cargo = models.CharField(max_length=20, choices=cargo_choices, null=True, blank=True)
    weight = models.FloatField(null=True, blank=True, default=0.0)
    start = models.ForeignKey('Station', on_delete=models.PROTECT, related_name='start')
    end = models.ForeignKey('Station', on_delete=models.PROTECT, related_name='end')
    start_date = models.DateField()
    end_date = models.DateField(null=True, blank=True)
    loaded = models.BooleanField(null=True, blank=True)
    previous = models.ForeignKey('self', on_delete=models.PROTECT, null=True, blank=True)
    locations = models.JSONField(null=True, blank=True)
    all_locations = models.JSONField(null=True, blank=True)

    def __str__(self) -> str:
        return f"{self.id} {self.wagon.num} {self.start} -> {self.end}"

    def set_ops_for_date(self, date: str, ops_point: Station, ops_date: str, ops: str) -> None:
        if self.locations is None:
            self.locations = {}
        if date in self.locations.keys():
            raise ValueError(f"Date {date} already exists")
        else:
            self.locations[date] = {'ops_station_id': ops_point.id, 'ops_date': ops_date, 'ops': ops}

    def get_ops_for_date(self, date: str) -> dict[Station, str, str] | None:

        if self.locations is None:
            self.locations = {}
            return None
        ops_point_id = self.locations[date]['ops_station_id']
        try:
            ops_point = Station.objects.get(id=ops_point_id)
            ops_date = self.locations[date]['ops_date']
            ops = self.locations[date]['ops']

        except Station.DoesNotExist:
            return None

        return {'ops_station': ops_point, 'ops_date': ops_date, 'ops': ops}

    @property
    def get_last_ops_id(self) -> str | None:
        if self.locations is None:
            self.locations = {}
            return None
        sorted_keys = sorted(self.locations.keys())
        last_key = sorted_keys[-1]
        return self.locations[last_key].get('ops_station_id', None)

    @property
    def get_last_ops_date(self) -> str | None:
        if self.locations is None or len(self.locations) == 0:
            self.locations = {}
            return None
        sorted_keys = list(sorted(self.locations.keys()))
        last_key = sorted_keys[-1]
        return self.locations[last_key].get('ops_date', None)

    def update_flat_locations(self) -> None:
        locations = {}
        for update, values in self.locations.items():
            locations[update] = values['ops_station_id']
        self.all_locations = locations

    def save(self, *args, **kwargs) -> None:
        self.update_flat_locations()
        super().save(*args, **kwargs)
