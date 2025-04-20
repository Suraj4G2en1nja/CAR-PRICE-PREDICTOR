from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# Load model and data
model = pickle.load(open("LinearRegressionModel.pkl", 'rb'))
car = pd.read_csv("cleaned car dataset.csv")

@app.route('/')
def index():
    companies = sorted(car['company'].unique())
    years = sorted(car['year'].unique(), reverse=True)
    fuel_types = sorted(car['fuel_type'].unique())

    companies.insert(0, "Select Company")
    years.insert(0, "Select Year")
    fuel_types.insert(0, "Select Fuel Type")

    return render_template('index.html', companies=companies, years=years, fuel_types=fuel_types, prediction=None)

@app.route('/get_car_models', methods=['POST'])
def get_car_models():
    company = request.form.get('company')
    models = sorted(car[car['company'] == company]['name'].unique())
    return jsonify(models)

@app.route('/predict', methods=['POST'])
def predict():
    company = request.form.get('company')
    car_model = request.form.get('car_model')
    year = int(request.form.get('year'))
    fuel_type = request.form.get('fuel_type')
    kms_driven = int(request.form.get('kms_driven'))

    # Prediction
    prediction = model.predict(pd.DataFrame([[car_model, company, year, kms_driven, fuel_type]],
                                            columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']))
    prediction_result = np.round(prediction[0], 2)
    if prediction_result < 0:
        prediction_result = 0.0

    companies = sorted(car['company'].unique())
    years = sorted(car['year'].unique(), reverse=True)
    fuel_types = sorted(car['fuel_type'].unique())

    companies.insert(0, "Select Company")
    years.insert(0, "Select Year")
    fuel_types.insert(0, "Select Fuel Type")

    return render_template('index.html', companies=companies, years=years, fuel_types=fuel_types,
                           prediction=f"Estimated Price: ₹ {prediction_result} lakhs")

if __name__ == "__main__":
    app.run(debug=True)

