from flask import request
from flask_restful import Resource

from app.services.classification import ClassificationService
from app.schemas.classification import ClassificationSchema

class ClassificationController(Resource):
    def post(self):
        data = ClassificationSchema.validate(request)
        classification = ClassificationService.classify(data)

        return { 'classification': classification }, 200
