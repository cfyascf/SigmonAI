class MlModel:
    def __init__(self):
        self.type = "ml"

    @staticmethod
    def get_instance():
        return MlModel()

    def predict(self, image):
        return "ml"