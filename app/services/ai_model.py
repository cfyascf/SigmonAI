class AiModel:
    def __init__(self):
        self.type = "ai"

    @staticmethod
    def get_instance():
        return AiModel()
    
    def predict(self, image):
        return "ai"