from app.data.enums import ModelType
from app.services.ai_model import AiModel
from app.services.ml_model import MlModel

class ClassificationService:
    __ai_model : AiModel
    __ml_model : MlModel

    def __init__(self):
        self.__ai_model = AiModel.get_instance()
        self.__ml_model = MlModel.get_instance()

    @staticmethod
    def classify(data):
        classifier = ClassificationService()
        classification = None

        print(type(classifier))
        print(type(classifier.__ai_model))
        print(type(classifier.__ml_model))

        if(data.model_type is ModelType.NEURAL_NETWORK):
            classification = classifier.__ai_model.predict(data.media)
        else:
            classification = classifier.__ml_model.predict(data.media)

        return classification