from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np


app = Flask(__name__)
model = pickle.load(open('Anomaly_Detection_model.pkl', 'rb'))
df = pd.read_csv('df_final.csv',delimiter=',',header='infer')    

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    #int_features = [float(x) for x in request.form.values()]
    #final_features = [np.array(int_features)]
    #prediction = model.predict(final_features)
    lst=[]
    for x in request.form.values():
       lst.append(x) 
    
    #input='(AO)Percentage SLAs Met={}, Ageing={}, Average Resolution Effort - Incidents={},      #Average Resolution Effort - Problems={}, Backlog Processing Efficiency - #Incidents={},Backlog Processing Efficiency - Problems={}, Delivered #Defects={}'.format(lst[0], lst[1],lst[2],lst[3],lst[4],lst[5],lst[6])            
    
    nearest_neighbor = model.kneighbors([lst], 1, return_distance=False)
    cluster=df.iloc[nearest_neighbor[0][0]+1].clusters
    anomaly=df.iloc[nearest_neighbor[0][0]+1].warning
    
    return render_template('index.html',  prediction_text='Prediction : {}, cluster:  {}, Index:  {}'.format(anomaly, cluster,nearest_neighbor+1))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    content = request.json
    nearest_neighbor = model.kneighbors([[100,0,38.11,2734.33,1.06,-0.32,0]], 1, return_distance=False)
    cluster=df.iloc[nearest_neighbor[0][0]+1].clusters
    anomaly=df.iloc[nearest_neighbor[0][0]+1].warning

    prediction_text='Anomaly {}, cluster {}, Index {}'.format(anomaly, cluster,nearest_neighbor+1)
    #return jsonify(output[0])
    return prediction_text

if __name__ == "__main__":
    app.run(debug=True)