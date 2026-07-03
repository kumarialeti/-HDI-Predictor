import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import pickle
import os

app = Flask(__name__, static_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static'), template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates'))
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'HDI.pkl')
model = pickle.load(open(model_path, 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/Prediction', methods=['POST', 'GET'])
def prediction():
    return render_template('indexnew.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve inputs from the form
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
    
    features_name = ['Country', 'Life expectancy', 'Mean years of schooling', 'Gross national income (GNI) per capita', 'Internet users']
    df = pd.DataFrame(features_value, columns=features_name)
    
    # Predict using the loaded model
    output = model.predict(df)
    
    # Handle single value output
    val = output[0] if isinstance(output, np.ndarray) and len(output.shape) == 1 else output[0][0]
    y_pred = round(val, 2)
    
    if y_pred <= 0.4:
        prediction_text = "Low HDI: " + str(y_pred)
    elif y_pred > 0.4 and y_pred <= 0.6:
        prediction_text = "Medium HDI: " + str(y_pred)
    elif y_pred > 0.6 and y_pred <= 0.8:
        prediction_text = "High HDI: " + str(y_pred)
    else:
        prediction_text = "Very High HDI: " + str(y_pred)
        
    return render_template('resultnew.html', prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
