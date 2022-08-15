import numpy as np
from flask import Flask,request,jsonify,render_template
import pickle

#creating flask app
app = Flask(__name__)

#load the model
# model = pickle.load(open("Models/model.pkl",'rb'))

@app.route('/')
def Home():
    return render_template('index.html')


@app.route('/predict',methods = ['POST'])
def predict():
    pass


if __name__ =="__main__":
    app.run(debug=True)