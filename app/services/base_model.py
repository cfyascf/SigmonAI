from abc import abstractmethod


class BaseModel:
    __path = r'C:\Users\disrct\Desktop\yasmim\SigmonAI\app\services\models\nn_singledb.sav'
    __categories = r'C:\Users\disrct\Desktop\yasmim\SigmonAI\app\services\models\categories.txt'

    def get_categories(self):
        with open(self.__categories, 'r') as file:
            alphabet_list = [line.strip() for line in file]

        return alphabet_list
    
    def get_path(self):
        return self.__path
    
    def get_categories(self):
        return self.__categories
    
    @abstractmethod
    def process_image(self, image):
        """This method should be overridden by subclasses."""
        pass

    @abstractmethod
    def predict(self, image):
        """This method should be overridden by subclasses."""
        pass