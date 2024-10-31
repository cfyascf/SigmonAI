from abc import abstractmethod
import os


class BaseModel:
    __categories = fr'{os.path.dirname(os.path.abspath(__file__))}\models\categories.txt'
    
    def get_categories(self):
        with open(self.__categories, 'r') as file:
            alphabet_list = [line.strip() for line in file]

        return alphabet_list
    
    @abstractmethod
    def process_image(self, image):
        """This method should be overridden by subclasses."""
        pass

    @abstractmethod
    def predict(self, image):
        """This method should be overridden by subclasses."""
        pass