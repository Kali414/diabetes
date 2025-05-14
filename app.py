# feature_names=['age','sex','bmi','bp','s1','s2','s3','s4','s5',  's6']
from flask import Flask, render_template, request,jsonify,url_for
from tensorflow.keras.models import load_model
import numpy as np
import joblib

model=load_model('model_diabetes.keras')
sc=joblib.load('standard_scalar.joblib')

app=Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    if(request.method=='POST'):
        age=int(request.form['age'])
        sex=int(request.form['sex'])
        bmi=float(request.form['bmi'])
        bp=float(request.form['bp'])
        s1=float(request.form['s1'])
        s2=float(request.form['s2'])
        s3=float(request.form['s3'])
        s4=float(request.form['s4'])
        s5=float(request.form['s5'])
        s6=float(request.form['s6'])

        input_features=[age,sex,bmi,bp,s1,s2,s3,s4,s5,s6]
        features=np.array(input_features).reshape(1,-1)
        scaled_features=sc.transform(features)

        prediction=model.predict(scaled_features)[0][0]
        print(prediction)

        return render_template('home.html',prediction_text='The patient is {}'.format(prediction))


    else:
        return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True,port=5000)