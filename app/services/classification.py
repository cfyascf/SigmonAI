from app.data.enums import ModelType
from app.services.nn_singledb_model import NNSingleDBModel
from app.services.random_forest_ml_model import RandomForestMLModel
from app.errors.app import AppError

class ClassificationService:
    __ai_model : NNSingleDBModel
    __ml_model : RandomForestMLModel

    def __init__(self):
        self.__ai_model = NNSingleDBModel.get_instance()
        self.__ml_model = RandomForestMLModel.get_instance()

    @staticmethod
    def classify(data):
        classifier = ClassificationService()
        classification = None

        model_type = data['model_type']
        media = data['file']

        match model_type:
            case ModelType.NEURAL_NETWORK:
                classification = classifier.__ai_model.predict(media)

            case ModelType.MACHINE_LEARNING:
                classification = classifier.__ml_model.predict(media)

            case _:
                raise AppError(403, "Not a valid model type.")

        return classification