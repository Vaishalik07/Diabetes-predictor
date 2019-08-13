from flask import Flask, request, jsonify, make_response
from flask_restplus import Api, Resource, fields
from sklearn.externals import joblib
import numpy as np
import sys
import tensorflow as tf

flask_app = Flask(__name__)
app = Api(app = flask_app,
		  version = "1.0",
		  title = "Diabetes Predictor",
		  description = "Predict the Diabetes status of a person")

name_space = app.namespace('prediction', description='Prediction APIs')
#preg,plas,pres,skin,insu,mass,pedi,age,
model = app.model('Prediction params',
				  {'pregVal': fields.Float(required = True,
				  							   description="Preg Val",
    					  				 	   help="Preg value cannot be blank"),
				  'plasVal': fields.Float(required = True,
				  							   description="plas Val",
    					  				 	   help="plas value cannot be blank"),
				  'presVal': fields.Float(required = True,
				  							description="pres Val",
    					  				 	help="Pres Val cannot be blank"),
				  'skinVal': fields.Float(required = True,
				  							description="skin Val",
    					  				 	help="Skin Val cannot be blank"),
                  'insuVal': fields.Float(required = True,
				  							description="insu Val",
    					  				 	help="Insu Val cannot be blank"),
                   'massVal': fields.Float(required = True,
  				  							description="mass Val",
      					  				 	help="mass Val cannot be blank"),
                   'pediVal': fields.Float(required = True,
  				  							description="pedi Val",
      					  				 	help="Pedi Val cannot be blank"),
                   'Age': fields.Float(required = True,
  				  							description="Age Val",
      					  				 	help="Age Val cannot be blank")})

classifier = joblib.load('classifier.joblib')
graph = tf.get_default_graph()

@name_space.route("/")
class MainClass(Resource):

	def options(self):
		response = make_response()
		response.headers.add("Access-Control-Allow-Origin", "*")
		response.headers.add('Access-Control-Allow-Headers', "*")
		response.headers.add('Access-Control-Allow-Methods', "*")
		return response

	@app.expect(model)
	def post(self):
		try:
			formData = request.json
			print(formData)
			data = [[int(val,10) for val in formData.values()]]
			global graph
			with graph.as_default():
				prediction = classifier.predict(data)
			types = { 0: "tested_negative", 1: "tested_positive"}
			response = jsonify({
				"statusCode": 200,
				"status": "Prediction made",
				"result": "Diabetes : " + types[prediction[0]]
				})
			response.headers.add('Access-Control-Allow-Origin', '*')
			return response
		except Exception as error:
			print(error)
			return jsonify({
				"statusCode": 500,
				"status": "Could not make prediction",
				"error": str(error)
			})