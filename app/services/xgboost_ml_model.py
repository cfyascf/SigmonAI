import io
import os
import joblib
import numpy as np
from pathlib import Path
import cv2
import mediapipe as mp

from app.services.base_model import BaseModel

class XGBoostMLModel(BaseModel):
    __path = fr'{os.path.dirname(os.path.abspath(__file__))}\models\ml_xgboost.pkl'
    __categories = fr'{os.path.dirname(os.path.abspath(__file__))}\models\ml_xgboost_label_encoder.pkl'
    __instance = None

    def __init__(self):
        super().__init__()
        self.__model = joblib.load(self.get_path())
        self.__categories = joblib.load(self.get_categories())

    @staticmethod
    def get_instance():
        if(XGBoostMLModel.__instance is None):
            xgboost = XGBoostMLModel()
            XGBoostMLModel.__instance = xgboost

        return XGBoostMLModel.__instance
    
    def process_image(self, image):
        path = Path("../Image")

        for file in path.glob("*"):
            if file.is_file():
                return str(file)
            
        return None
    
    def extract_hand_landmarks(self, image):
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
        
        image_data = image.read()
        nparr = np.frombuffer(image_data, np.uint8)
        mat_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        results = hands.process(cv2.cvtColor(mat_image, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            landmarks = results.multi_hand_landmarks[0]
            return [coord for lm in landmarks.landmark for coord in (lm.x, lm.y)]
        
        return None
    
    def predict(self, image):
        
        img_data = self.extract_hand_landmarks(image)

        if img_data is not None:
            img_data = [img_data]
            predict = self.__model.predict(img_data)
            predict_label = self.__categories.inverse_transform(predict)
            return predict_label[0]
        else:
            return None
        
    