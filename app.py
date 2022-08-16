import numpy as np
from flask import Flask,request,jsonify,render_template
import pickle as p
import joblib
import sys
import pandas as pd

#creating flask app
app = Flask(__name__)

#load the model
# model = pickle.load(open("Models/model.pkl",'rb'))

@app.route('/')
def Home():
    return render_template('index.html')


@app.route('/predict',methods = ['POST'])
def predict():
    if request.method == "POST": 

        # getting user input from HTML form
        district = request.form["district_name"]
        road_class = request.form["road_class"]
        traffctl = request.form["traffic_control"]
        visibility = request.form["visib_name"]
        light = request.form["light_name"]
        impacttype=request.form["impact_type"]
        age=request.form['age']
        road_condition = request.form["road_condition"]
        #default values based on most frequent occurence in database
        month=8
        day=20
        time=1700
        street1='YONGE ST'
        street2='LAWRENCE AVE E'
        latitude=43.740245
        longitude=-79.251190
        loccoord='Intersection'
        invtype='Driver'
        injury='None'
        pedestrian=1
        cyclist=1
        automobile=0
        motorcycle=1
        truck=1
        trsn_city_veh=1
        emerg_veh=0
        passenger=0
        speeding=2
        ag_driv=1
        redlight=1
        alcohol=0
        disability=0
        
        query=pd.DataFrame([[month, day, time, street1, street2, 
                                   latitude, longitude, road_class, district, 
                                   loccoord, traffctl, visibility, light, 
                                   road_condition, impacttype, invtype, age, 
                                   injury, automobile, ag_driv, pedestrian, 
                                   cyclist, motorcycle, truck, trsn_city_veh, 
                                   emerg_veh, passenger, speeding, disability,
                                   alcohol, redlight]], columns=model_columns)
        query=query.reindex()
        
        prediction=lr.predict(query)
        
        print(prediction)
        
        if prediction[0]==0:
            outcome="Fatal"
        else:
            outcome="Non-fatal"
        
        return render_template('index.html', pred=outcome)


if __name__ =="__main__":
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12345 # If you don't provide any port the port will be set to 12345

    lr = joblib.load('Data/bestmodel_lr.pkl') # Load "model.pkl"
    print ('Model loaded')
    model_columns = joblib.load('Data/model_columns.pkl') # Load "model_columns.pkl"
    print ('Model columns loaded')
    
    app.run(port=port, debug=True)
    app.run(debug=True)