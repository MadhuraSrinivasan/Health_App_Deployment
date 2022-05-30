
import pickle
import joblib
from flask import Flask,render_template, url_for,request,jsonify
import numpy as np
app = Flask(__name__)
model=pickle.load(open('finalized_model.sav','rb'))
heart_model = pickle.load(open('heart_model.sav','rb'))

@app.route('/')
@app.route('/home')
def home():
    return render_template('base.html')

@app.route('/heart')
def test1():
    return render_template('heart.html',title='Heart')

@app.route('/diabetes')
def contact():
    return render_template('diabetes.html',title='Diabetes')



@app.route('/help')
def help():
    return render_template('help.html')

@app.route('/predict', methods=['POST'])
def predict():
   features= [int(x) for x in request.form.values()]
   print(features)
   final_features = [np.array(features)]
  
   prediction= model.predict(final_features)
   output = round(prediction[0],2)
   if output==0.0:
       return render_template('diabetes.html',predict_result='Hurrayy!! You are not diagonised with diabetes.')
   else:
        return render_template('diabetes.html',predict_result='You are diagonised with Diabetes. Kindly consult your doctor.')


@app.route('/predictheart',methods=['POST'])
def predict_heart():
    feature = [a for a in request.form.values()]
    print(feature)
    final_feature = [np.array(feature)]
    #print(final_feature)
    heart_predict = heart_model.predict(final_feature)
    res = round(heart_predict[0],2)
    if res==0.0:
        return render_template('heart.html',predicted_output = 'Your heart is totally healthy !! Maintain the same way.')
    else:
        return render_template('heart.html',predicted_output = 'Your heart needs doctor attention. ')


   

if __name__=="__main__":
    app.run(debug=True)


