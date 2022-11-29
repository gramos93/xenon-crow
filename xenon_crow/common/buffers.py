from random import choices
from torch.utils.data import Dataset


class BasicBuffer(Dataset):
    def __init__(self, data_handler, max_size=1e5, batch_size=32) -> None:
        super(BasicBuffer).__init__()

        self.max_size = max_size
        self.batch_size = batch_size
        self.__buffer = []
        self.__data_handler = data_handler
    
    def sample(self):

        batch = choices(self.__buffer, k=self.batch_size)
        return [self.__data_handler.fetch(samples) for samples in batch]

    def push(self, data):

        if len(self.__buffer) == self.max_size:
            del self.__buffer[0]
        
        self.__buffer.append(data)

    def ready(self):
        return len(self.__buffer) >= self.batch_size
