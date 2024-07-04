from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load the models
#with open('linear_model.pkl', 'rb') as file:
#    linear_model = pickle.load(file)

#with open('decision_tree_model.pkl', 'rb') as file:
    #decision_tree_model = pickle.load(file)

with open('random_forest_model.pkl', 'rb') as file:
    random_forest_model = pickle.load(file)

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    year = data['year']
    month = data['month']

    # Prepare the input data for the model
    input_data = np.array([[year, month]])

    # Choose the best model for prediction
    prediction = random_forest_model.predict(input_data)[0]

    return jsonify({'prediction': prediction})


if __name__ == '__main__':
    host = '0.0.0.0'
    port = 5000

    print(f"Starting Flask app on {host}:{port}")
    app.run(debug=True, host=host, port=port)


