from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync


def send_progress_update(message: str) -> None:
    channel_layer = get_channel_layer()
    async_to_sync(channel_layer.group_send)("messages", {"type": "send.progress", "message": message})
