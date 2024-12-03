from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
import pickle
import numpy as np

model_path = 'rf_model.pkl'
vectorizer_path = 'vectorizer.pkl'

with open(model_path, 'rb') as file:
    model = pickle.load(file)
with open(vectorizer_path, 'rb') as file:
    vectorizer = pickle.load(file)

app = Flask(__name__)
app.secret_key = 'your_secret_key'        #Resolve form resubmission problem where this line is required for flashing messages

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    message = request.form['mixedInput']   

    message_vectorized = vectorizer.transform([message])  
    
    prediction = model.predict(message_vectorized)
    output = 'a Spam!' if prediction[0] == 1 else 'Not a Spam'

    flash('Message is {}'.format(output))                       #Resolves form resubmission problem
    return redirect(url_for('index'))                           #Resolves form resubmission problem

    #return render_template('index.html', prediction_text='Message is {}'.format(output))   #'Conform form resubmission' is to be expected if used


if __name__ == "__main__":
    app.run(debug=True)