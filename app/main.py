from flask import Flask, request
from os.path import dirname, join
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

app= Flask(__name__)
@app.route('/')
def index():
  return "<h1>Welcome to CodingX</h1>"


@app.route("/predict", methods=['POST'])
def predict():
  if request.method == 'POST':
    message = request.form['message']
    data = [message]
  return message

