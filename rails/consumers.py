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
        # Handle view_status_update message
        message = event['message']
        # Your processing logic for view_status_update message
        
        # Send the message to the endpoint page
        text_data = json.dumps({
            'type': 'status_update',
            'message': message
        })
        await self.send(text_data=text_data)


    async def receive(self, text_data):

        # Receive message from WebSocket
        text_data_json = json.loads(text_data)
        message = text_data_json['message']

        # Send message to room group
        #await self.channel_layer.group_send(
        #    self.group_name,
        #    {
        #        'type': 'view_status_update',
        #        'message': message
        #    }
        #)
        await self.view_status_update(text_data_json)

    async def chat_message(self, event):
        # Send message to WebSocket
        message = event['message']
        await self.send(text_data=json.dumps({
            'message': message
        }))