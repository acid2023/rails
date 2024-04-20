import json

from channels.generic.websocket import AsyncWebsocketConsumer


class ViewStatusConsumer(AsyncWebsocketConsumer):
    channel_name = 'view_status'

    async def connect(self):
        self.group_name = 'view_status'
        # Join room group
        await self.channel_layer.group_add(self.group_name, self.channel_name)

        await self.accept()

    async def disconnect(self, close_code):
        # Leave room group
        await self.channel_layer.group_discard(self.group_name, self.channel_name)

    async def view_status_update(self, event):
        message = event['message']
        text_data = json.dumps({
            'type': 'status_update',
            'message': message
        })
        await self.send(text_data=text_data)

    async def receive(self, text_data):
        text_data_json = json.loads(text_data)
        await self.view_status_update(text_data_json)

    async def chat_message(self, event):
        # Send message to WebSocket
        message = event['message']
        await self.send(text_data=json.dumps({
            'message': message
        }))

    async def send_message(self, message):
        self.send(text_data=message)
