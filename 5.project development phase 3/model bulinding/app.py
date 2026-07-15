from flask import Flask, request, render_template
import pickle
import numpy as np

# 1. Initialize the Flask engine app
app = Flask(__name__)

# 2. Open our saved AI model from Epic 4 
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# 3. Define what happens when someone loads our website homepage
@app.route('/')
def home():
    return render_template('index.html')

# 4. Define what happens when someone clicks the "Predict" button on the webpage
@app.route('/predict', methods=['POST'])
def predict():
    # Grab the 7 numbers typed into the HTML inputs by the user
    input_features = [
        float(request.form['nitrogen']),
        float(request.form['phosphorous']),
        float(request.form['potassium']),
        float(request.form['temperature']),
        float(request.form['humidity']),
        float(request.form['ph']),
        float(request.form['rainfall'])
    ]
    
    # Format the numbers into an array so our model can process them
    final_features = [np.array(input_features)]
    
    # Run the machine learning model prediction
    prediction = model.predict(final_features)
    
    output = prediction[0]
    
    # Send the answer back to index.html using the 'prediction_text' variable
    return render_template('index.html', prediction_text=f'The suggested crop for given climatic condition is: {output}')

# 5. Start the local server
if __name__ == "__main__":
    app.run(debug=True)
