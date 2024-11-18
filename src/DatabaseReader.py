from abc import abstractmethod


class DatabaseReader:
    @abstractmethod
    def read_table(self):
        pass