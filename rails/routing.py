from channels.routing import ProtocolTypeRouter, URLRouter
from channels.auth import AuthMiddlewareStack

from django.urls import path

from .consumers import ViewStatusConsumer

websocket_urlpatterns = [
    # Define WebSocket URL patterns directly without the path function
    path('ws/view_status/', ViewStatusConsumer()),
    
]

application = ProtocolTypeRouter({
    'websocket': AuthMiddlewareStack(
        URLRouter(
            websocket_urlpatterns
        )
    ),
})