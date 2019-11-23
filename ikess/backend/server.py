from flask import Flask, request
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)

todos = {}


class DigitClassifier(Resource):
    def get(self, frame):
        data = request.form['data']

        return {'data': data}


api.add_resource(DigitClassifier, '/')


if __name__ == '__main__':
    app.run(debug=True)
