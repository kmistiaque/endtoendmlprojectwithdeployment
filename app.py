import pickle
import numpy as np
from flask import Flask, request, app, jsonify, url_for, render_template

app = Flask(__name__)

# Load the model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Define constants for feature engineering
mean_latitude = 35.63186143410853  
mean_longitude = -119.56970445736432
min_latitude = 32.54  

@app.route('/')
def home():
    return render_template('predicthomeprice.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    """
    API endpoint to predict house price using JSON data.
    """
    # Get data from the request
    data = request.json['data']
    print("Raw input data:", data)

    # Input features
    input_features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']

    # Convert input data to a dictionary
    inputs = {feature: data[feature] for feature in input_features}

    # Engineer additional features
    rooms_to_bedrooms = inputs['AveRooms'] / inputs['AveBedrms'] if inputs['AveBedrms'] != 0 else 0
    population_density = inputs['Population'] / (inputs['Latitude'] - min_latitude) if (inputs['Latitude'] - min_latitude) != 0 else 0
    distance_from_center = np.sqrt((inputs['Latitude'] - mean_latitude)**2 + (inputs['Longitude'] - mean_longitude)**2)

    print("Engineered features:", rooms_to_bedrooms, population_density, distance_from_center)

    # Prepare data for scaling
    final_features = [
        inputs['MedInc'],
        inputs['AveRooms'],
        inputs['HouseAge'],
        inputs['Latitude'],
        inputs['Longitude'],
        rooms_to_bedrooms,
        population_density,
        distance_from_center
    ]

    print("Final input data:", final_features)

    # Scale the data
    scaled_data = scaler.transform(np.array(final_features).reshape(1, -1))
    print("Scaled input data:", scaled_data)

    # Make prediction
    prediction = model.predict(scaled_data)[0]
    print("Predicted house price:", prediction)

    # Return the prediction as JSON
    return jsonify({'predicted_price': prediction * 100000})

@app.route('/predict', methods=['POST'])
def predict():
    """
    Web form submission to predict house price.
    """
    # Extract data from form
    data = [float(x) for x in request.form.values()]
    input_features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
    inputs = dict(zip(input_features, data))

    # Engineer additional features
    rooms_to_bedrooms = inputs['AveRooms'] / inputs['AveBedrms'] if inputs['AveBedrms'] != 0 else 0
    population_density = inputs['Population'] / (inputs['Latitude'] - min_latitude) if (inputs['Latitude'] - min_latitude) != 0 else 0
    distance_from_center = np.sqrt((inputs['Latitude'] - mean_latitude)**2 + (inputs['Longitude'] - mean_longitude)**2)

    # Prepare data for scaling
    final_features = [
        inputs['MedInc'],
        inputs['AveRooms'],
        inputs['HouseAge'],
        inputs['Latitude'],
        inputs['Longitude'],
        rooms_to_bedrooms,
        population_density,
        distance_from_center
    ]

    # Scale the data
    scaled_data = scaler.transform(np.array(final_features).reshape(1, -1))
    print("Scaled input data:", scaled_data)

    # Make prediction
    prediction = model.predict(scaled_data)[0]

    # Render the prediction on the webpage
    return render_template(
        "predicthomeprice.html",
        prediction_text=f"The House price prediction is ${prediction * 100000:.2f}"
    )

if __name__ == "__main__":
    app.run(debug=True)
