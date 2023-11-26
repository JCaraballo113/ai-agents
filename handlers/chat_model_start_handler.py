from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema.messages import BaseMessage


class ChatModelStartHandler(BaseCallbackHandler):
    def on_chat_model_start(self, serialized, messages, **kwargs):
        print(messages)
