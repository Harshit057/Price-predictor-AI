import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

# Use the correct path to load the model
model_path = "../boston_house_price_model.pkl"
model = joblib.load(model_path)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    prediction = model.predict([data["features"]])
    return jsonify({"prediction": prediction.tolist()})

if __name__ == "__main__":
    app.run(debug=True)
