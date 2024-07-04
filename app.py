from flask import Flask, request, render_template
import pickle as pk
import numpy as np

# Create flask application
app = Flask(__name__,template_folder="templates",static_folder="static",static_url_path="/")

# load model
ModelClassifier = pk.load(open('rf_tuned_model.sav','rb'))

# Create home route
@app.route('/', methods = ['GET', 'POST'])
def home():
    return render_template('index.html', result = False)


# Create predict page
@app.route('/predict', methods = ['POST'])
def predict():
    # Retrieve form data and convert to floats
    ph_value = float(request.form.get('ph'))
    hardness_value = float(request.form.get('hardness'))
    solids_value = float(request.form.get('solids'))
    solids_value = np.log(solids_value)
    chloramines_value = float(request.form.get('chloramines'))
    sulfate_value = float(request.form.get('sulfate'))
    conductivity_value = float(request.form.get('conductivity'))
    organic_carbon_value = float(request.form.get('organic_carbon'))
    trihalomethanes_value = float(request.form.get('trihalomethanes'))
    turbidity_value = float(request.form.get('turbidity'))

    # Combine the inputs into a numpy array
    input_features = np.array([[ph_value, hardness_value, solids_value, 
                                chloramines_value, sulfate_value, conductivity_value, 
                                organic_carbon_value, trihalomethanes_value, turbidity_value]])

    # Make prediction using the loaded model
    prediced_value = ModelClassifier.predict(input_features)[0]

    # Determine message based on prediction
    if prediced_value:
        message = "The water is safe for human consumption (Potable)."
    else:
        message = "The water is not safe for human consumption (Not Potable)."

    return render_template('index.html', result = message)

if __name__ == "__main__":
    app.run(debug=True)



