import io
import joblib
import numpy as np
from PIL import Image

from app.services.base_model import BaseModel

class NNSingleDBModel(BaseModel):
    __instance = None

    def __init__(self):
        super().__init__()
        self.__model = joblib.load(self.get_path())

    @staticmethod
    def get_instance():
        if(NNSingleDBModel.__instance is None):
            nn = NNSingleDBModel()
            NNSingleDBModel.__instance = nn

        return NNSingleDBModel.__instance
    
    def process_image(self, image):
        image_stream = io.BytesIO(image.read())
        img_opened = Image.open(image_stream)
        img_array = np.asarray(img_opened)

        return np.expand_dims(img_array, axis=0)
    
    def predict(self, image):
        expanded_img = self.process_image(image)
        pred = self.__model.predict(expanded_img)
        answear = self.get_categories()[np.argmax(pred)]

        return answear