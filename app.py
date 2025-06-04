from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

import joblib
model = joblib.load('D:\projects\Space ship prediction\data\spaceship-titanic\model2.pkl')


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            # Collect all form values
            data = {
                'HomePlanet': request.form['HomePlanet'],
                'CryoSleep': request.form['CryoSleep'] == 'True',
                'Cabin': request.form['Cabin'],
                'Destination': request.form['Destination'],
                'Age': float(request.form['Age']),
                'VIP': request.form['VIP'] == 'True',
                'RoomService': float(request.form['RoomService']),
                'FoodCourt': float(request.form['FoodCourt']),
                'ShoppingMall': float(request.form['ShoppingMall']),
                'Spa': float(request.form['Spa']),
                'VRDeck': float(request.form['VRDeck']),
            }

            # Convert to DataFrame (same structure as training)
            input_df = pd.DataFrame([data])

            # Make prediction
            prediction = model.predict(input_df)[0]
        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
