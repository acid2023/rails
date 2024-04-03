from django.contrib import admin

from rails.models import Station, Road, Wagon, Route


class AdminRoad(admin.ModelAdmin):
    list_display = ['name']


class AdminStation(admin.ModelAdmin):
    list_display = ['name', 'code', 'road', 'coords_lat', 'coords_lon', 'point']
    search_fields = ['name', 'code']


class AdminRoute(admin.ModelAdmin):
    list_display = ['start', 'end', 'start_date', 'end_date', 'locations', 'all_locations']


admin.site.register(Station, AdminStation)
admin.site.register(Road, AdminRoad)
admin.site.register(Wagon)
admin.site.register(Route, AdminRoute)
