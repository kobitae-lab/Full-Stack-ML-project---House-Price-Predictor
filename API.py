from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from pymongo import MongoClient

app = Flask(__name__)
CORS(app)

client = MongoClient("mongodb://localhost:27017")
db = client["house_predictor"]
collection = db["predictions"]

scaler = joblib.load('scaler.pkl')
model = tf.keras.models.load_model("house_model.keras")

# confirm API running
@app.route('/')
def home():
    data = {
        "Test1" : "Text",
        "Test2" : "Is this visible"
    }
    return jsonify(data)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        arr = [data["MedInc"], data["HouseAge"], data["AveRooms"], data["AveBedrms"], data["Population"], data["AveOccup"], data["Latitude"], data["Longitude"]]
        np_data = np.array(arr)
        np_data = np_data.reshape(1, -1)
        scaled_data = scaler.transform(np_data)
    
        prediction = float(model.predict(scaled_data).flatten()[0] * 100000)

        return jsonify({
            "Predicted_Price" : prediction,
            "Input_Features" : data
        })
    except Exception as e:
        return jsonify({"error" :  str(e)}), 400
         



@app.route('/predictions', methods=['GET'])
def get_predictions():
    try:
        docs = collection.find().sort("_id", -1).limit(10)

        predictions = []
        for doc in docs:
            doc["_id"] = str(doc["_id"])
            predictions.append(doc)

        return jsonify({
            "predictions": predictions,
            "count": len(predictions)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, port=5000)