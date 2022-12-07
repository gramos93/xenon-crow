from collections import deque
from random import choices


class RandomBuffer(object):
    def __init__(self, data_handler, max_size: int = 1e5, batch_size: int = 32) -> None:
        super(RandomBuffer).__init__()
        self.batch_size = batch_size
        self.__buffer = deque(maxlen=max_size)
        self.__data_handler = data_handler

    def sample(self):
        batch = choices(self.__buffer, k=self.batch_size)
        return self.__data_handler(batch)

    def push(self, data):
        self.__buffer.append(data)

    def ready(self):
        return len(self.__buffer) >= self.batch_size


class ReinforceBuffer(object):
    def __init__(self, data_handler) -> None:
        super(ReinforceBuffer).__init__()
        self.__buffer = list()
        self.__data_handler = data_handler

    def __iter__(self):
        for replay in self.__buffer:
            yield self.__data_handler(replay)

    def push(self, data):
        self.__buffer.append(data)

    def __len__(self):
        return len(self.__buffer)

    def reset(self):
        del self.__buffer[:]
