"""
URL configuration for rails project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from channels.routing import URLRouter, ProtocolTypeRouter
from channels.auth import AuthMiddlewareStack

from .views import road_area_detail, upload_xls_file, process_files, create_rails_models, update_rails_models, make_prediction, load_data
from .consumers import ViewStatusConsumer

websocket_urlpatterns = [
    # Define WebSocket URL patterns directly without the path function
    path('ws/view_status/', ViewStatusConsumer),
    ]
urlpatterns = [
    path('process-files/', process_files, name='process_files'),
    path('admin/', admin.site.urls),
    path('map/<int:area_id>/', road_area_detail),
    path('upload/', upload_xls_file, name='upload_xls_file'),
    path('load_data/', load_data, name='load_data'),
    path('create_models/', create_rails_models),
    path('update_models/', update_rails_models),
    path('make_prediction/', make_prediction),
    *websocket_urlpatterns,
]

application = ProtocolTypeRouter({
    'websocket': AuthMiddlewareStack(
        URLRouter(
            websocket_urlpatterns
        )
    ),
})

