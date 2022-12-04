from collections import deque
from random import choices


class BasicBuffer(object):
    def __init__(self, data_handler, max_size:int=1e5, batch_size:int=32) -> None:
        super(BasicBuffer).__init__()

        self.batch_size = batch_size
        self.__buffer = deque(maxlen=int(max_size))
        self.__data_handler = data_handler
    
    def sample(self):
        batch = choices(self.__buffer, k=self.batch_size)
        return self.__data_handler(batch)

    def push(self, data):
        self.__buffer.append(data)

    def ready(self):
        return len(self.__buffer) >= self.batch_size
