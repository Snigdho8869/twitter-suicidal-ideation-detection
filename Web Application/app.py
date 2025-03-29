from flask import Flask, request, jsonify, render_template, redirect, url_for
from flask_mail import Mail, Message
import smtplib
import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import pickle

app = Flask(__name__, template_folder='templates', static_folder='static')

model = load_model('Twitter_Suicidal_Ideation_Detection_GRU.h5')

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/index.html')
def index():
    return render_template('index.html')

@app.route('/contact.html')
def contact():
    return render_template('contact.html')

@app.route('/about.html')
def about():
    return render_template('about.html')

@app.route('/resources.html')
def resources():
    return render_template('resources.html')

@app.route('/suicide-ideation', methods=['POST'])
def predict():
    text = request.json['text']
    if not text:
        predictionText = 'Please Enter Some Text'
        response = jsonify({'predictionText': predictionText})
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    
    twt = tokenizer.texts_to_sequences([text])
    twt = pad_sequences(twt, maxlen=60, dtype='int32')
    
    predicted = model.predict(twt, batch_size=1, verbose=True)
    
    if np.argmax(predicted) == 0:
        predictionText = 'Potential Suicide Post'
    elif np.argmax(predicted) == 1:
        predictionText = 'Non Suicide Post'
    else:
        predictionText = 'Unknown'
    
    response = jsonify({'prediction': float(predicted[0][0]), 'predictionText': predictionText})
    response.headers.add('Access-Control-Allow-Origin', '*')
    
    return response

@app.route('/send-email', methods=['POST'])
def send_email():
    name = request.form['name']
    email = request.form['email']
    message = request.form['message']
    
    subject = 'Contact Form Submission from ' + name
    body = 'Name: ' + name + '\nEmail: ' + email + '\nMessage: ' + message
    
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login('zahidulislam2225@gmail.com', 'your_password_here')
    
    server.sendmail('zahidulislam2225@gmail.com', 'rafin3600@gmail.com', subject + '\n\n' + body)
    server.quit()
    
    return render_template('thank-you.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)