from watson_machine_learning_client import WatsonMachineLearningAPIClient
from flask import request
from flask import jsonify
import os
from flask import Flask,render_template,request,jsonify
import io
import xarray as xr
import datetime
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
from numpy import concatenate
import urllib3, requests, json



#
# 1.  Fill in wml_credentials.
#
wrl = {
  "apikey": "-ShzG2q7k3seI7uXt8GsHnd7e1KQKj2qblE8D_uxjjHp",
  "instance_id": "f1e3dce4-3cb6-430a-bb2a-81969e0c8d48",
  "url": "https://eu-gb.ml.cloud.ibm.com"
}

client = WatsonMachineLearningAPIClient( wrl )

#
# 2.  Fill in one or both of these:
#     - model_deployment_endpoint_url
#     - function_deployment_endpoint_url
#
model_deployment_endpoint_url    = 'https://eu-gb.ml.cloud.ibm.com/v3/wml_instances/f1e3dce4-3cb6-430a-bb2a-81969e0c8d48/deployments/ea6c1a0c-a30c-433d-b99f-3a136f6bcef8/online';
function_deployment_endpoint_url = "";


STATIC_FOLDER = 'templates/assets'
app = Flask(__name__,static_folder=STATIC_FOLDER)


    

# On IBM Cloud Cloud Foundry, get the port number from the environment variable PORT
# When running this app on the local machine, default the port to 8000
port = int(os.getenv('PORT', 8080))




def data_gathering(latitude, longitude):
    
    nc = xr.open_dataset('../app/download.nc')
    collected_lat = latitude
    collected_lon = longitude
    lat = nc.latitude

    for i in range(len(lat)):
        if (lat[i] == collected_lat):
            print(i)
            m = i
    lon = nc.longitude
    for j in range(len(lon)):
        if (lon[j] == collected_lon):
            print(j)
            n = j
    time = nc.time
    list = ['u100', 'v100', 't2m', 'i10fg', 'sp']

    df = pd.DataFrame()

    for o in range(0, len(list)):
        Values = []
        for p in range(0, len(time)):
            a = nc[list[o]][p][m][n]
            a = np.array(a)
            a = a.item()
            Values.append(a)
        df[list[o]] = pd.Series(Values)

    df['Time'] = nc.time
    df.set_index('Time', inplace=True)
    df=df.fillna(df.mean())
    df1 = pd.DataFrame()
    df1['Air_Density'] = df.sp / (287.058 * df.t2m)
    df1['Wind_Speed'] = np.sqrt((df.u100 ** 2) + (df.v100 ** 2))
    return df1


def data_preperation(df1):
    df = df1[(len(df1) - 1):]
    return df


def Test_data_preperation(df, df1):
    result = []
    actual = []
    values = df1.values
    # ensure all data is float
    values = values.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    test_values = df.values
    # ensure all data is float
    test_values = test_values.astype('float32')
    # normalize features
    test_scaled = scaler.transform(test_values)
    test_X = test_scaled
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    scoring_endpoint=model_deployment_endpoint_url
    payload={"fields":["Air_Density","Wind_Speed"], "values": [[[str(test_X[0][0][0]),str(test_X[0][0][1])]]]}
    for i in range(120):
            y_prediction = client.deployments.score(scoring_endpoint,payload)
            result.append([y_prediction['values'][0][0]])
            result[i] = scaler.inverse_transform(result[i])
            actual.append(result[i])
            scaled = scaler.transform(result[i])
            result_next = scaled.reshape((scaled.shape[0], 1, scaled.shape[1]))
            payload={"fields":["Air_Density","Wind_Speed"], "values": [[[str(result_next[0][0][0]),str(result_next[0][0][1])]]]}
    data = pd.DataFrame(np.concatenate(result), columns=['Air Density', 'Wind Speed'])
    lists = [50, 70, 80, 100]
    cp = 0.59
    for i in range(len(lists)):
        A = 3.14 * (lists[i] ** 2)
        data['Energy_' + str(lists[i]) + 'm(MW)'] = (0.5 * (data['Air Density']) * A * (
                    data['Wind Speed'] ** 3) * cp) / 1000000
    Turbines=[10,15,20,30]
    Energy=['Energy_50m(MW)','Energy_70m(MW)','Energy_80m(MW)','Energy_100m(MW)']
    for j in range(len(Turbines)):
      for i in range(len(Energy)):
            data['No_of_Turbines'+str(Turbines[j])+'T'+str(Energy[i])]=Turbines[j]*data[Energy[i]]
    return data

@app.route('/Home.html')
def home():
    return render_template('Home.html')
@app.route('/Predict.html',methods=['POST','GET'])
def Future():
    return render_template('Predict.html')
@app.route('/Team.html')
def Team():
    return render_template('Team.html')
@app.route('/Contact Us.html')
def Contact():
    return render_template('Contact Us.html')
@app.route('/Land.html')
def Land():
    return render_template('Land.html')
@app.route('/Login.html')
def Login():
    return render_template('Login.html')
@app.route('/Registration.html')
def Registration():
    return render_template('Registration.html')
@app.route('/predict', methods=['POST', 'GET'])

def predict():
    int_features=[float(x) for x in request.form.values()]
    final=[np.array(int_features)]
    print(final)
    latitude,longitude,time,radius,turbines=final[0][0],final[0][1],final[0][2],final[0][3],final[0][4]
    print(latitude,longitude)
    df1 = data_gathering(round(latitude), round(longitude))
    df = data_preperation(df1)
    print(df)
    data = Test_data_preperation(df, df1)
    rf48=pd.DataFrame()
    Variables=data.columns
    for i in range(2,len(Variables)):
      rf48[Variables[i]]=[data[Variables[i]][0]+data[Variables[i]][1]]

    rf72=pd.DataFrame()
    Variables=data.columns
    for i in range(2,len(Variables)):
      rf72[Variables[i]]=[rf48[Variables[i]][0]+data[Variables[i]][2]]
    
    rf1m=pd.DataFrame()
    Variables=data.columns
    for i in range(2,len(Variables)):
      a=[]
      for j in range(29):
        a.append(data[Variables[i]][j]+data[Variables[i]][j+1])
      b=np.sum(a)
      rf1m[Variables[i]]=[b]
    
    rf4m=pd.DataFrame()
    Variables=data.columns
    for i in range(2,len(Variables)):
      a=[]
      for j in range(119):
        a.append(data[Variables[i]][j]+data[Variables[i]][j+1])
      b=np.sum(a)
      rf4m[Variables[i]]=[b]
      
    if (time==24.0 and radius==50.0 and turbines==10.0):
        rf24ft=pd.DataFrame()
        rf24ft['Air Density']=[data['Air Density'][0]]
        rf24ft['Wind Speed(24hr)']=[data['Wind Speed'][0]]
        rf24ft['Energy_50m(MW)']=[data['Energy_50m(MW)'][0]]
        rf24ft['10Turbines(MW)']=[data['No_of_Turbines10TEnergy_50m(MW)'][0]]
        return render_template('Predict.html',tables=[rf24ft.to_html(classes='data')], titles=df.columns.values)
    elif (time==24.0 and radius==50.0 and turbines==15.0):
        rf24ff=pd.DataFrame()
        rf24ff['Air Density']=[data['Air Density'][0]]
        rf24ff['Wind Speed(24hr)']=[data['Wind Speed'][0]]
        rf24ff['Energy_50m(MW)']=[data['Energy_50m(MW)'][0]]
        rf24ff['15Turbines(MW)']=[data['No_of_Turbines15TEnergy_50m(MW)'][0]]
        return render_template('Predict.html',tables=[rf24ff.to_html(classes='data')], titles=df.columns.values)
    elif (time==24.0 and radius==50.0 and turbines==20.0):
        rf24ftw=pd.DataFrame()
        rf24ftw['Air Density']=[data['Air Density'][0]]
        rf24ftw['Wind Speed(24hr)']=[data['Wind Speed'][0]]
        rf24ftw['Energy_50m(MW)']=[data['Energy_50m(MW)'][0]]
        rf24ftw['20Turbines(MW)']=[data['No_of_Turbines20TEnergy_50m(MW)'][0]]
        return render_template('Predict.html',tables=[rf24ftw.to_html(classes='data')], titles=df.columns.values)
    elif (time==24.0 and radius==50.0 and turbines==30.0):
        rf24fth=pd.DataFrame()
        rf24fth['Air Density']=[data['Air Density'][0]]
        rf24fth['Wind Speed(24hr)']=[data['Wind Speed'][0]]
        rf24fth['Energy_50m(MW)']=[data['Energy_50m(MW)'][0]]
        rf24fth['30Turbines(MW)']=[data['No_of_Turbines30TEnergy_50m(MW)'][0]]
        return render_template('Predict.html',tables=[rf24fth.to_html(classes='data')], titles=df.columns.values)
    elif (time==24.0 and radius==70.0 and turbines==10.0):
        rf24st=pd.DataFrame()
        rf24st['Air Density']=[data['Air Density'][0]]
        rf24st['Wind Speed(24hr)']=[data['Wind Speed'][0]]
        rf24st['Energy_70m(MW)']=[data['Energy_70m(MW)'][0]]
        rf24st['10Turbines(MW)']=[data['No_of_Turbines10TEnergy_70m(MW)'][0]]
        return render_template('Predict.html',tables=[rf24st.to_html(classes='data')], titles=df.columns.values)
    elif (time==24.0 and radius==70.0 and turbines==15.0):
        rf24sf=pd.DataFrame()
        rf24sf['Air Density']=[data['Air Density'][0]]
        rf24sf['Wind Speed(24hr)']=[data['Wind Speed'][0]]
        rf24sf['Energy_70m(MW)']=[data['Energy_70m(MW)'][0]]
        rf24sf['15Turbines(MW)']=[data['No_of_Turbines15TEnergy_70m(MW)'][0]]
        return render_template('Predict.html',tables=[rf24sf.to_html(classes='data')], titles=df.columns.values)
    elif (time==24.0 and radius==70.0 and turbines==20.0):
        rf24stw=pd.DataFrame()
        rf24stw['Air Density']=[data['Air Density'][0]]
        rf24stw['Wind Speed(24hr)']=[data['Wind Speed'][0]]
        rf24stw['Energy_70m(MW)']=[data['Energy_70m(MW)'][0]]
        rf24stw['20Turbines(MW)']=[data['No_of_Turbines20TEnergy_70m(MW)'][0]]
        return render_template('Predict.html',tables=[rf24stw.to_html(classes='data')], titles=df.columns.values)
    elif (time==24.0 and radius==70.0 and turbines==30.0):
        rf24sth=pd.DataFrame()
        rf24sth['Air Density']=[data['Air Density'][0]]
        rf24sth['Wind Speed(24hr)']=[data['Wind Speed'][0]]
        rf24sth['Energy_70m(MW)']=[data['Energy_70m(MW)'][0]]
        rf24sth['30Turbines(MW)']=[data['No_of_Turbines30TEnergy_70m(MW)'][0]]
        return render_template('Predict.html',tables=[rf24sth.to_html(classes='data')], titles=df.columns.values)
    elif (time==24.0 and radius==80.0 and turbines==10.0):
        rf24et=pd.DataFrame()
        rf24et['Air Density']=[data['Air Density'][0]]
        rf24et['Wind Speed(24hr)']=[data['Wind Speed'][0]]
        rf24et['Energy_80m(MW)']=[data['Energy_80m(MW)'][0]]
        rf24et['10Turbines(MW)']=[data['No_of_Turbines10TEnergy_80m(MW)'][0]]
        return render_template('Predict.html',tables=[rf24et.to_html(classes='data')], titles=df.columns.values)
    elif (time==24.0 and radius==80.0 and turbines==15.0):
        rf24ef=pd.DataFrame()
        rf24ef['Air Density']=[data['Air Density'][0]]
        rf24ef['Wind Speed(24hr)']=[data['Wind Speed'][0]]
        rf24ef['Energy_80m(MW)']=[data['Energy_80m(MW)'][0]]
        rf24ef['15Turbines(MW)']=[data['No_of_Turbines15TEnergy_80m(MW)'][0]]
        return render_template('Predict.html',tables=[rf24ef.to_html(classes='data')], titles=df.columns.values)
    elif (time==24.0 and radius==80.0 and turbines==20.0):
        rf24etw=pd.DataFrame()
        rf24etw['Air Density']=[data['Air Density'][0]]
        rf24etw['Wind Speed(24hr)']=[data['Wind Speed'][0]]
        rf24etw['Energy_80m(MW)']=[data['Energy_80m(MW)'][0]]
        rf24etw['20Turbines(MW)']=[data['No_of_Turbines20TEnergy_80m(MW)'][0]]
        return render_template('Predict.html',tables=[rf24etw.to_html(classes='data')], titles=df.columns.values)
    elif (time==24.0 and radius==80.0 and turbines==30.0):
        rf24eth=pd.DataFrame()
        rf24eth['Air Density']=[data['Air Density'][0]]
        rf24eth['Wind Speed(24hr)']=[data['Wind Speed'][0]]
        rf24eth['Energy_80m(MW)']=[data['Energy_80m(MW)'][0]]
        rf24eth['30Turbines(MW)']=[data['No_of_Turbines30TEnergy_80m(MW)'][0]]
        return render_template('Predict.html',tables=[rf24eth.to_html(classes='data')], titles=df.columns.values)
    elif (time==24.0 and radius==100.0 and turbines==10.0):
        rf24ht=pd.DataFrame()
        rf24ht['Air Density']=[data['Air Density'][0]]
        rf24ht['Wind Speed(24hr)']=[data['Wind Speed'][0]]
        rf24ht['Energy_100m(MW)']=[data['Energy_100m(MW)'][0]]
        rf24ht['10Turbines(MW)']=[data['No_of_Turbines10TEnergy_100m(MW)'][0]]
        return render_template('Predict.html',tables=[rf24ht.to_html(classes='data')], titles=df.columns.values)
    elif (time==24.0 and radius==100.0 and turbines==15.0):
        rf24hf=pd.DataFrame()
        rf24hf['Air Density']=[data['Air Density'][0]]
        rf24hf['Wind Speed(24hr)']=[data['Wind Speed'][0]]
        rf24hf['Energy_100m(MW)']=[data['Energy_100m(MW)'][0]]
        rf24hf['15Turbines(MW)']=[data['No_of_Turbines15TEnergy_100m(MW)'][0]]
        return render_template('Predict.html',tables=[rf24hf.to_html(classes='data')], titles=df.columns.values)
    elif (time==24.0 and radius==100.0 and turbines==20.0):
        rf24htw=pd.DataFrame()
        rf24htw['Air Density']=[data['Air Density'][0]]
        rf24htw['Wind Speed(24hr)']=[data['Wind Speed'][0]]
        rf24htw['Energy_100m(MW)']=[data['Energy_100m(MW)'][0]]
        rf24htw['20Turbines(MW)']=[data['No_of_Turbines20TEnergy_100m(MW)'][0]]
        return render_template('Predict.html',tables=[rf24htw.to_html(classes='data')], titles=df.columns.values)
    elif (time==24.0 and radius==100.0 and turbines==30.0):
        rf24hth=pd.DataFrame()
        rf24hth['Air Density']=[data['Air Density'][0]]
        rf24hth['Wind Speed(24hr)']=[data['Wind Speed'][0]]
        rf24hth['Energy_100m(MW)']=[data['Energy_100m(MW)'][0]]
        rf24hth['30Turbines(MW)']=[data['No_of_Turbines30TEnergy_100m(MW)'][0]]
        return render_template('Predict.html',tables=[rf24eth.to_html(classes='data')], titles=df.columns.values)
    elif (time==48.0 and radius==50.0 and turbines==10.0):
        rf48ft=pd.DataFrame()
        rf48ft['Air Density']=[data['Air Density'][1]]
        rf48ft['Wind Speed(48hr)']=[data['Wind Speed'][1]]
        rf48ft['Energy_50m(MW)']=[rf48['Energy_50m(MW)'][0]]
        rf48ft['10Turbines(MW)']=[rf48['No_of_Turbines10TEnergy_50m(MW)'][0]]
        return render_template('Predict.html',tables=[rf48ft.to_html(classes='data')], titles=df.columns.values)
    elif (time==48.0 and radius==50.0 and turbines==15.0):
        rf48ff=pd.DataFrame()
        rf48ff['Air Density']=[data['Air Density'][1]]
        rf48ff['Wind Speed(48hr)']=[data['Wind Speed'][1]]
        rf48ff['Energy_50m(MW)']=[rf48['Energy_50m(MW)'][0]]
        rf48ff['15Turbines(MW)']=[rf48['No_of_Turbines15TEnergy_50m(MW)'][0]]
        return render_template('Predict.html',tables=[rf48ff.to_html(classes='data')], titles=df.columns.values)
    elif (time==48.0 and radius==50.0 and turbines==20.0):
        rf48ftw=pd.DataFrame()
        rf48ftw['Air Density']=[data['Air Density'][1]]
        rf48ftw['Wind Speed(48hr)']=[data['Wind Speed'][1]]
        rf48ftw['Energy_50m(MW)']=[rf48['Energy_50m(MW)'][0]]
        rf48ftw['20Turbines(MW)']=[rf48['No_of_Turbines20TEnergy_50m(MW)'][0]]
        return render_template('Predict.html',tables=[rf48ftw.to_html(classes='data')], titles=df.columns.values)
    elif (time==48.0 and radius==50.0 and turbines==30.0):
        rf48fth=pd.DataFrame()
        rf48fth['Air Density']=[data['Air Density'][1]]
        rf48fth['Wind Speed(48hr)']=[data['Wind Speed'][1]]
        rf48fth['Energy_50m(MW)']=[rf48['Energy_50m(MW)'][0]]
        rf48fth['30Turbines(MW)']=[rf48['No_of_Turbines30TEnergy_50m(MW)'][0]]
        return render_template('Predict.html',tables=[rf24fth.to_html(classes='data')], titles=df.columns.values)
    elif (time==48.0 and radius==70.0 and turbines==10.0):
        rf48st=pd.DataFrame()
        rf48st['Air Density']=[data['Air Density'][1]]
        rf48st['Wind Speed(48hr)']=[data['Wind Speed'][1]]
        rf48st['Energy_70m(MW)']=[rf48['Energy_70m(MW)'][0]]
        rf48st['10Turbines(MW)']=[rf48['No_of_Turbines10TEnergy_70m(MW)'][0]]
        return render_template('Predict.html',tables=[rf48st.to_html(classes='data')], titles=df.columns.values)
    elif (time==48.0 and radius==70.0 and turbines==15.0):
        rf48sf=pd.DataFrame()
        rf48sf['Air Density']=[data['Air Density'][1]]
        rf48sf['Wind Speed(48hr)']=[data['Wind Speed'][1]]
        rf48sf['Energy_70m(MW)']=[rf48['Energy_70m(MW)'][0]]
        rf48sf['15Turbines(MW)']=[rf48['No_of_Turbines15TEnergy_70m(MW)'][0]]
        return render_template('Predict.html',tables=[rf48sf.to_html(classes='data')], titles=df.columns.values)
    elif (time==48.0 and radius==70.0 and turbines==20.0):
        rf48stw=pd.DataFrame()
        rf48stw['Air Density']=[data['Air Density'][1]]
        rf48stw['Wind Speed(48hr)']=[data['Wind Speed'][1]]
        rf48stw['Energy_70m(MW)']=[rf48['Energy_70m(MW)'][0]]
        rf48stw['20Turbines(MW)']=[rf48['No_of_Turbines20TEnergy_70m(MW)'][0]]
        return render_template('Predict.html',tables=[rf48stw.to_html(classes='data')], titles=df.columns.values)
    elif (time==48.0 and radius==70.0 and turbines==30.0):
        rf48sth=pd.DataFrame()
        rf48sth['Air Density']=[data['Air Density'][1]]
        rf48sth['Wind Speed(48hr)']=[data['Wind Speed'][1]]
        rf48sth['Energy_70m(MW)']=[rf48['Energy_70m(MW)'][0]]
        rf48sth['30Turbines(MW)']=[rf48['No_of_Turbines30TEnergy_70m(MW)'][0]]
        return render_template('Predict.html',tables=[rf24sth.to_html(classes='data')], titles=df.columns.values)
    elif (time==48.0 and radius==80.0 and turbines==10.0):
        rf48et=pd.DataFrame()
        rf48et['Air Density(48hr)']=[data['Air Density'][1]]
        rf48et['Wind Speed']=[data['Wind Speed'][1]]
        rf48et['Energy_80m(MW)']=[rf48['Energy_80m(MW)'][0]]
        rf48et['10Turbines(MW)']=[rf48['No_of_Turbines10TEnergy_80m(MW)'][0]]
        return render_template('Predict.html',tables=[rf48et.to_html(classes='data')], titles=df.columns.values)
    elif (time==48.0 and radius==80.0 and turbines==15.0):
        rf48ef=pd.DataFrame()
        rf48ef['Air Density']=[data['Air Density'][1]]
        rf48ef['Wind Speed(48hr)']=[data['Wind Speed'][1]]
        rf48ef['Energy_80m(MW)']=[rf48['Energy_80m(MW)'][0]]
        rf48ef['15Turbines(MW)']=[rf48['No_of_Turbines15TEnergy_80m(MW)'][0]]
        return render_template('Predict.html',tables=[rf48ef.to_html(classes='data')], titles=df.columns.values)
    elif (time==48.0 and radius==80.0 and turbines==20.0):
        rf48etw=pd.DataFrame()
        rf48etw['Air Density']=[data['Air Density'][1]]
        rf48etw['Wind Speed(48hr)']=[data['Wind Speed'][1]]
        rf48etw['Energy_80m(MW)']=[rf48['Energy_80m(MW)'][0]]
        rf48etw['20Turbines(MW)']=[rf48['No_of_Turbines20TEnergy_80m(MW)'][0]]
        return render_template('Predict.html',tables=[rf48etw.to_html(classes='data')], titles=df.columns.values)
    elif (time==48.0 and radius==80.0 and turbines==30.0):
        rf48eth=pd.DataFrame()
        rf48eth['Air Density']=[data['Air Density'][1]]
        rf48eth['Wind Speed(48hr)']=[data['Wind Speed'][1]]
        rf48eth['Energy_80m(MW)']=[rf48['Energy_80m(MW)'][0]]
        rf48eth['30Turbines(MW)']=[rf48['No_of_Turbines30TEnergy_80m(MW)'][0]]
        return render_template('Predict.html',tables=[rf48eth.to_html(classes='data')], titles=df.columns.values)
    elif (time==48.0 and radius==100.0 and turbines==10.0):
        rf48ht=pd.DataFrame()
        rf48ht['Air Density']=[data['Air Density'][1]]
        rf48ht['Wind Speed(48hr)']=[data['Wind Speed'][1]]
        rf48ht['Energy_100m(MW)']=[rf48['Energy_100m(MW)'][0]]
        rf48ht['10Turbines(MW)']=[rf48['No_of_Turbines10TEnergy_100m(MW)'][0]]
        return render_template('Predict.html',tables=[rf48ht.to_html(classes='data')], titles=df.columns.values)
    elif (time==48.0 and radius==100.0 and turbines==15.0):
        rf48hf=pd.DataFrame()
        rf48hf['Air Density']=[data['Air Density'][1]]
        rf48hf['Wind Speed(48hr)']=[data['Wind Speed'][1]]
        rf48hf['Energy_100m(MW)']=[rf48['Energy_100m(MW)'][0]]
        rf48hf['15Turbines(MW)']=[rf48['No_of_Turbines15TEnergy_100m(MW)'][0]]
        return render_template('Predict.html',tables=[rf48hf.to_html(classes='data')], titles=df.columns.values)
    elif (time==48.0 and radius==100.0 and turbines==20.0):
        rf48htw=pd.DataFrame()
        rf48htw['Air Density']=[data['Air Density'][1]]
        rf48htw['Wind Speed(48hr)']=[data['Wind Speed'][1]]
        rf48htw['Energy_100m(MW)']=[rf48['Energy_100m(MW)'][0]]
        rf48htw['20Turbines(MW)']=[rf48['No_of_Turbines20TEnergy_100m(MW)'][0]]
        return render_template('Predict.html',tables=[rf48htw.to_html(classes='data')], titles=df.columns.values)
    elif (time==48.0 and radius==100.0 and turbines==30.0):
        rf48hth=pd.DataFrame()
        rf48hth['Air Density']=[data['Air Density'][1]]
        rf48hth['Wind Speed(48hr)']=[data['Wind Speed'][1]]
        rf48hth['Energy_100m(MW)']=[rf48['Energy_100m(MW)'][0]]
        rf48hth['30Turbines(MW)']=[rf48['No_of_Turbines30TEnergy_100m(MW)'][0]]
        return render_template('Predict.html',tables=[rf48hth.to_html(classes='data')], titles=df.columns.values)
    elif (time==72.0 and radius==50.0 and turbines==10.0):
        rf72ft=pd.DataFrame()
        rf72ft['Air Density']=[data['Air Density'][2]]
        rf72ft['Wind Speed(72hr)']=[data['Wind Speed'][2]]
        rf72ft['Energy_50m(MW)']=[rf72['Energy_50m(MW)'][0]]
        rf72ft['10Turbines(MW)']=[rf72['No_of_Turbines10TEnergy_50m(MW)'][0]]
        return render_template('Predict.html',tables=[rf72ft.to_html(classes='data')], titles=df.columns.values)
    elif (time==72.0 and radius==50.0 and turbines==15.0):
        rf72ff=pd.DataFrame()
        rf72ff['Air Density']=[data['Air Density'][2]]
        rf72ff['Wind Speed(72hr)']=[data['Wind Speed'][2]]
        rf72ff['Energy_50m(MW)']=[rf72['Energy_50m(MW)'][0]]
        rf72ff['15Turbines(MW)']=[rf72['No_of_Turbines15TEnergy_50m(MW)'][0]]
        return render_template('Predict.html',tables=[rf72ff.to_html(classes='data')], titles=df.columns.values)
    elif (time==72.0 and radius==50.0 and turbines==20.0):
        rf72ftw=pd.DataFrame()
        rf72ftw['Air Density']=[data['Air Density'][2]]
        rf72ftw['Wind Speed(72hr)']=[data['Wind Speed'][2]]
        rf72ftw['Energy_50m(MW)']=[rf72['Energy_50m(MW)'][0]]
        rf72ftw['20Turbines(MW)']=[rf72['No_of_Turbines20TEnergy_50m(MW)'][0]]
        return render_template('Predict.html',tables=[rf72ftw.to_html(classes='data')], titles=df.columns.values)
    elif (time==72.0 and radius==50.0 and turbines==30.0):
        rf72fth=pd.DataFrame()
        rf72fth['Air Density']=[data['Air Density'][2]]
        rf72fth['Wind Speed(72hr)']=[data['Wind Speed'][2]]
        rf72fth['Energy_50m(MW)']=[rf72['Energy_50m(MW)'][0]]
        rf72fth['30Turbines(MW)']=[rf72['No_of_Turbines30TEnergy_50m(MW)'][0]]
        return render_template('Predict.html',tables=[rf72fth.to_html(classes='data')], titles=df.columns.values)
    elif (time==72.0 and radius==70.0 and turbines==10.0):
        rf72st=pd.DataFrame()
        rf72st['Air Density']=[data['Air Density'][2]]
        rf72st['Wind Speed(72hr)']=[data['Wind Speed'][2]]
        rf72st['Energy_70m(MW)']=[rf72['Energy_70m(MW)'][0]]
        rf72st['10Turbines(MW)']=[rf72['No_of_Turbines10TEnergy_70m(MW)'][0]]
        return render_template('Predict.html',tables=[rf72st.to_html(classes='data')], titles=df.columns.values)
    elif (time==72.0 and radius==70.0 and turbines==15.0):
        rf72sf=pd.DataFrame()
        rf72sf['Air Density']=[data['Air Density'][2]]
        rf72sf['Wind Speed(72hr)']=[data['Wind Speed'][2]]
        rf72sf['Energy_70m(MW)']=[rf72['Energy_70m(MW)'][0]]
        rf72sf['15Turbines(MW)']=[rf72['No_of_Turbines15TEnergy_70m(MW)'][0]]
        return render_template('Predict.html',tables=[rf72sf.to_html(classes='data')], titles=df.columns.values)
    elif (time==72.0 and radius==70.0 and turbines==20.0):
        rf72stw=pd.DataFrame()
        rf72stw['Air Density']=[data['Air Density'][2]]
        rf72stw['Wind Speed(72hr)']=[data['Wind Speed'][2]]
        rf72stw['Energy_70m(MW)']=[rf72['Energy_70m(MW)'][0]]
        rf72stw['20Turbines(MW)']=[rf72['No_of_Turbines20TEnergy_70m(MW)'][0]]
        return render_template('Predict.html',tables=[rf72stw.to_html(classes='data')], titles=df.columns.values)
    elif (time==72.0 and radius==70.0 and turbines==30.0):
        rf72sth=pd.DataFrame()
        rf72sth['Air Density']=[data['Air Density'][2]]
        rf72sth['Wind Speed(72hr)']=[data['Wind Speed'][2]]
        rf72sth['Energy_70m(MW)']=[rf72['Energy_70m(MW)'][0]]
        rf72sth['30Turbines(MW)']=[rf72['No_of_Turbines30TEnergy_70m(MW)'][0]]
        return render_template('Predict.html',tables=[rf72sth.to_html(classes='data')], titles=df.columns.values)    
    elif (time==72.0 and radius==80.0 and turbines==10.0):
        rf72et=pd.DataFrame()
        rf72et['Air Density']=[data['Air Density'][2]]
        rf72et['Wind Speed(72hr)']=[data['Wind Speed'][2]]
        rf72et['Energy_80m(MW)']=[rf72['Energy_80m(MW)'][0]]
        rf72et['10Turbines(MW)']=[rf72['No_of_Turbines10TEnergy_80m(MW)'][0]]
        return render_template('Predict.html',tables=[rf72et.to_html(classes='data')], titles=df.columns.values)
    elif (time==72.0 and radius==80.0 and turbines==15.0):
        rf72ef=pd.DataFrame()
        rf72ef['Air Density']=[data['Air Density'][2]]
        rf72ef['Wind Speed(72hr)']=[data['Wind Speed'][2]]
        rf72ef['Energy_80m(MW)']=[rf72['Energy_80m(MW)'][0]]
        rf72ef['15Turbines(MW)']=[rf72['No_of_Turbines15TEnergy_80m(MW)'][0]]
        return render_template('Predict.html',tables=[rf72ef.to_html(classes='data')], titles=df.columns.values)
    elif (time==72.0 and radius==80.0 and turbines==20.0):
        rf72etw=pd.DataFrame()
        rf72etw['Air Density']=[data['Air Density'][2]]
        rf72etw['Wind Speed(72hr)']=[data['Wind Speed'][2]]
        rf72etw['Energy_80m(MW)']=[rf72['Energy_80m(MW)'][0]]
        rf72etw['20Turbines(MW)']=[rf72['No_of_Turbines20TEnergy_80m(MW)'][0]]
        return render_template('Predict.html',tables=[rf72etw.to_html(classes='data')], titles=df.columns.values)
    elif (time==72.0 and radius==80.0 and turbines==30.0):
        rf72eth=pd.DataFrame()
        rf72eth['Air Density']=[data['Air Density'][2]]
        rf72eth['Wind Speed(72hr)']=[data['Wind Speed'][2]]
        rf72eth['Energy_80m(MW)']=[rf72['Energy_80m(MW)'][0]]
        rf72eth['30Turbines(MW)']=[rf72['No_of_Turbines30TEnergy_80m(MW)'][0]]
        return render_template('Predict.html',tables=[rf72eth.to_html(classes='data')], titles=df.columns.values) 
    elif (time==72.0 and radius==100.0 and turbines==10.0):
        rf72ht=pd.DataFrame()
        rf72ht['Air Density']=[data['Air Density'][2]]
        rf72ht['Wind Speed(72hr)']=[data['Wind Speed'][2]]
        rf72ht['Energy_100m(MW)']=[rf72['Energy_100m(MW)'][0]]
        rf72ht['10Turbines(MW)']=[rf72['No_of_Turbines10TEnergy_100m(MW)'][0]]
        return render_template('Predict.html',tables=[rf72ht.to_html(classes='data')], titles=df.columns.values)
    elif (time==72.0 and radius==100.0 and turbines==15.0):
        rf72hf=pd.DataFrame()
        rf72hf['Air Density']=[data['Air Density'][2]]
        rf72hf['Wind Speed(72hr)']=[data['Wind Speed'][2]]
        rf72hf['Energy_100m(MW)']=[rf72['Energy_100m(MW)'][0]]
        rf72hf['15Turbines(MW)']=[rf72['No_of_Turbines15TEnergy_100m(MW)'][0]]
        return render_template('Predict.html',tables=[rf72hf.to_html(classes='data')], titles=df.columns.values)
    elif (time==72.0 and radius==100.0 and turbines==20.0):
        rf72htw=pd.DataFrame()
        rf72htw['Air Density']=[data['Air Density'][2]]
        rf72htw['Wind Speed(72hr)']=[data['Wind Speed'][2]]
        rf72htw['Energy_100m(MW)']=[rf72['Energy_100m(MW)'][0]]
        rf72htw['20Turbines(MW)']=[rf72['No_of_Turbines20TEnergy_100m(MW)'][0]]
        return render_template('Predict.html',tables=[rf72htw.to_html(classes='data')], titles=df.columns.values)
    elif (time==72.0 and radius==100.0 and turbines==30.0):
        rf72hth=pd.DataFrame()
        rf72hth['Air Density']=[data['Air Density'][2]]
        rf72hth['Wind Speed(72hr)']=[data['Wind Speed'][2]]
        rf72hth['Energy_100m(MW)']=[rf72['Energy_100m(MW)'][0]]
        rf72hth['30Turbines(MW)']=[rf72['No_of_Turbines30TEnergy_100m(MW)'][0]]
        return render_template('Predict.html',tables=[rf72hth.to_html(classes='data')], titles=df.columns.values)
    elif (time==1.0 and radius==50.0 and turbines==10.0):
        rf1mft=pd.DataFrame()
        rf1mft['Air Density']=[data['Air Density'][29]]
        rf1mft['Wind Speed(1M)']=[data['Wind Speed'][29]]
        rf1mft['Energy_50m(MW)']=[rf1m['Energy_50m(MW)'][0]]
        rf1mft['10Turbines(MW)']=[rf1m['No_of_Turbines10TEnergy_50m(MW)'][0]]
        return render_template('Predict.html',tables=[rf1mft.to_html(classes='data')], titles=df.columns.values)
    elif (time==1.0 and radius==50.0 and turbines==15.0):
        rf1mff=pd.DataFrame()
        rf1mff['Air Density']=[data['Air Density'][29]]
        rf1mff['Wind Speed(1M)']=[data['Wind Speed'][29]]
        rf1mff['Energy_50m(MW)']=[rf1m['Energy_50m(MW)'][0]]
        rf1mff['15Turbines(MW)']=[rf1m['No_of_Turbines15TEnergy_50m(MW)'][0]]
        return render_template('Predict.html',tables=[rf1mff.to_html(classes='data')], titles=df.columns.values)
    elif (time==1.0 and radius==50.0 and turbines==20.0):
        rf1mftw=pd.DataFrame()
        rf1mftw['Air Density']=[data['Air Density'][29]]
        rf1mftw['Wind Speed(1M)']=[data['Wind Speed'][29]]
        rf1mftw['Energy_50m(MW)']=[rf1m['Energy_50m(MW)'][0]]
        rf1mftw['20Turbines(MW)']=[rf1m['No_of_Turbines20TEnergy_50m(MW)'][0]]
        return render_template('Predict.html',tables=[rf1mftw.to_html(classes='data')], titles=df.columns.values)
    elif (time==1.0 and radius==50.0 and turbines==30.0):
        rf1mfth=pd.DataFrame()
        rf1mfth['Air Density']=[data['Air Density'][29]]
        rf1mfth['Wind Speed(1M)']=[data['Wind Speed'][29]]
        rf1mfth['Energy_50m(MW)']=[rf1m['Energy_50m(MW)'][0]]
        rf1mfth['30Turbines(MW)']=[rf1m['No_of_Turbines30TEnergy_50m(MW)'][0]]
        return render_template('Predict.html',tables=[rf1mfth.to_html(classes='data')], titles=df.columns.values)
    elif (time==1.0 and radius==70.0 and turbines==10.0):
        rf1mst=pd.DataFrame()
        rf1mst['Air Density']=[data['Air Density'][29]]
        rf1mst['Wind Speed(1M)']=[data['Wind Speed'][29]]
        rf1mst['Energy_70m(MW)']=[rf1m['Energy_70m(MW)'][0]]
        rf1mst['10Turbines(MW)']=[rf1m['No_of_Turbines10TEnergy_70m(MW)'][0]]
        return render_template('Predict.html',tables=[rf1mst.to_html(classes='data')], titles=df.columns.values)
    elif (time==1.0 and radius==70.0 and turbines==15.0):
        rf1msf=pd.DataFrame()
        rf1msf['Air Density']=[data['Air Density'][29]]
        rf1msf['Wind Speed(1M)']=[data['Wind Speed'][29]]
        rf1msf['Energy_70m(MW)']=[rf1m['Energy_70m(MW)'][0]]
        rf1msf['15Turbines(MW)']=[rf1m['No_of_Turbines15TEnergy_70m(MW)'][0]]
        return render_template('Predict.html',tables=[rf1msf.to_html(classes='data')], titles=df.columns.values)
    elif (time==1.0 and radius==70.0 and turbines==20.0):
        rf1mstw=pd.DataFrame()
        rf1mstw['Air Density']=[data['Air Density'][29]]
        rf1mstw['Wind Speed(1M)']=[data['Wind Speed'][29]]
        rf1mstw['Energy_70m(MW)']=[rf1m['Energy_70m(MW)'][0]]
        rf1mstw['20Turbines(MW)']=[rf1m['No_of_Turbines20TEnergy_70m(MW)'][0]]
        return render_template('Predict.html',tables=[rf1mstw.to_html(classes='data')], titles=df.columns.values)
    elif (time==1.0 and radius==70.0 and turbines==30.0):
        rf1msth=pd.DataFrame()
        rf1msth['Air Density']=[data['Air Density'][29]]
        rf1msth['Wind Speed(1M)']=[data['Wind Speed'][29]]
        rf1msth['Energy_70m(MW)']=[rf1m['Energy_70m(MW)'][0]]
        rf1msth['30Turbines(MW)']=[rf1m['No_of_Turbines30TEnergy_70m(MW)'][0]]
        return render_template('Predict.html',tables=[rf1msth.to_html(classes='data')], titles=df.columns.values)
    elif (time==1.0 and radius==80.0 and turbines==10.0):
        rf1met=pd.DataFrame()
        rf1met['Air Density']=[data['Air Density'][29]]
        rf1met['Wind Speed(1M)']=[data['Wind Speed'][29]]
        rf1met['Energy_80m(MW)']=[rf1m['Energy_80m(MW)'][0]]
        rf1met['10Turbines(MW)']=[rf1m['No_of_Turbines10TEnergy_80m(MW)'][0]]
        return render_template('Predict.html',tables=[rf1met.to_html(classes='data')], titles=df.columns.values)
    elif (time==1.0 and radius==80.0 and turbines==15.0):
        rf1mef=pd.DataFrame()
        rf1mef['Air Density']=[data['Air Density'][29]]
        rf1mef['Wind Speed(1M)']=[data['Wind Speed'][29]]
        rf1mef['Energy_80m(MW)']=[rf1m['Energy_80m(MW)'][0]]
        rf1mef['15Turbines(MW)']=[rf1m['No_of_Turbines15TEnergy_80m(MW)'][0]]
        return render_template('Predict.html',tables=[rf1mef.to_html(classes='data')], titles=df.columns.values)
    elif (time==1.0 and radius==80.0 and turbines==20.0):
        rf1metw=pd.DataFrame()
        rf1metw['Air Density']=[data['Air Density'][29]]
        rf1metw['Wind Speed(1M)']=[data['Wind Speed'][29]]
        rf1metw['Energy_80m(MW)']=[rf1m['Energy_80m(MW)'][0]]
        rf1metw['20Turbines(MW)']=[rf1m['No_of_Turbines20TEnergy_80m(MW)'][0]]
        return render_template('Predict.html',tables=[rf1metw.to_html(classes='data')], titles=df.columns.values)
    elif (time==1.0 and radius==80.0 and turbines==30.0):
        rf1meth=pd.DataFrame()
        rf1meth['Air Density']=[data['Air Density'][29]]
        rf1meth['Wind Speed(1M)']=[data['Wind Speed'][29]]
        rf1meth['Energy_80m(MW)']=[rf1m['Energy_80m(MW)'][0]]
        rf1meth['30Turbines(MW)']=[rf1m['No_of_Turbines30TEnergy_80m(MW)'][0]]
        return render_template('Predict.html',tables=[rf1meth.to_html(classes='data')], titles=df.columns.values)
    elif (time==1.0 and radius==100.0 and turbines==10.0):
        rf1mht=pd.DataFrame()
        rf1mht['Air Density']=[data['Air Density'][29]]
        rf1mht['Wind Speed(1M)']=[data['Wind Speed'][29]]
        rf1mht['Energy_100m(MW)']=[rf1m['Energy_100m(MW)'][0]]
        rf1mht['10Turbines(MW)']=[rf1m['No_of_Turbines10TEnergy_100m(MW)'][0]]
        return render_template('Predict.html',tables=[rf1mht.to_html(classes='data')], titles=df.columns.values)
    elif (time==1.0 and radius==100.0 and turbines==15.0):
        rf1mhf=pd.DataFrame()
        rf1mhf['Air Density']=[data['Air Density'][29]]
        rf1mhf['Wind Speed(1M)']=[data['Wind Speed'][29]]
        rf1mhf['Energy_100m(MW)']=[rf1m['Energy_100m(MW)'][0]]
        rf1mhf['15Turbines(MW)']=[rf1m['No_of_Turbines15TEnergy_100m(MW)'][0]]
        return render_template('Predict.html',tables=[rf1mhf.to_html(classes='data')], titles=df.columns.values)
    elif (time==1.0 and radius==100.0 and turbines==20.0):
        rf1mhtw=pd.DataFrame()
        rf1mhtw['Air Density']=[data['Air Density'][29]]
        rf1mhtw['Wind Speed(1M)']=[data['Wind Speed'][29]]
        rf1mhtw['Energy_100m(MW)']=[rf1m['Energy_100m(MW)'][0]]
        rf1mhtw['20Turbines(MW)']=[rf1m['No_of_Turbines20TEnergy_100m(MW)'][0]]
        return render_template('Predict.html',tables=[rf1mhtw.to_html(classes='data')], titles=df.columns.values)
    elif (time==1.0 and radius==100.0 and turbines==30.0):
        rf1mhth=pd.DataFrame()
        rf1mhth['Air Density']=[data['Air Density'][29]]
        rf1mhth['Wind Speed(1M)']=[data['Wind Speed'][29]]
        rf1mhth['Energy_100m(MW)']=[rf1m['Energy_100m(MW)'][0]]
        rf1mhth['30Turbines(MW)']=[rf1m['No_of_Turbines30TEnergy_100m(MW)'][0]]
        return render_template('Predict.html',tables=[rf1mhth.to_html(classes='data')], titles=df.columns.values)
    elif (time==4.0 and radius==50.0 and turbines==10.0):
        rf4mft=pd.DataFrame()
        rf4mft['Air Density']=[data['Air Density'][119]]
        rf4mft['Wind Speed(4M)']=[data['Wind Speed'][119]]
        rf4mft['Energy_50m(MW)']=[rf4m['Energy_50m(MW)'][0]]
        rf4mft['10Turbines(MW)']=[rf4m['No_of_Turbines10TEnergy_50m(MW)'][0]]
        return render_template('Predict.html',tables=[rf4mft.to_html(classes='data')], titles=df.columns.values)
    elif (time==4.0 and radius==50.0 and turbines==15.0):
        rf4mff=pd.DataFrame()
        rf4mff['Air Density']=[data['Air Density'][119]]
        rf4mff['Wind Speed(4M)']=[data['Wind Speed'][119]]
        rf4mff['Energy_50m(MW)']=[rf4m['Energy_50m(MW)'][0]]
        rf4mff['15Turbines(MW)']=[rf4m['No_of_Turbines15TEnergy_50m(MW)'][0]]
        return render_template('Predict.html',tables=[rf4mff.to_html(classes='data')], titles=df.columns.values)
    elif (time==4.0 and radius==50.0 and turbines==20.0):
        rf4mftw=pd.DataFrame()
        rf4mftw['Air Density']=[data['Air Density'][119]]
        rf4mftw['Wind Speed(4M)']=[data['Wind Speed'][119]]
        rf4mftw['Energy_50m(MW)']=[rf4m['Energy_50m(MW)'][0]]
        rf4mftw['20Turbines(MW)']=[rf4m['No_of_Turbines20TEnergy_50m(MW)'][0]]
        return render_template('Predict.html',tables=[rf4mftw.to_html(classes='data')], titles=df.columns.values)
    elif (time==4.0 and radius==50.0 and turbines==30.0):
        rf4mfth=pd.DataFrame()
        rf4mfth['Air Density']=[data['Air Density'][119]]
        rf4mfth['Wind Speed(4M)']=[data['Wind Speed'][119]]
        rf4mfth['Energy_50m(MW)']=[rf4m['Energy_50m(MW)'][0]]
        rf4mfth['30Turbines(MW)']=[rf4m['No_of_Turbines30TEnergy_50m(MW)'][0]]
        return render_template('Predict.html',tables=[rf4mfth.to_html(classes='data')], titles=df.columns.values)
    elif (time==4.0 and radius==70.0 and turbines==10.0):
        rf4mst=pd.DataFrame()
        rf4mst['Air Density']=[data['Air Density'][119]]
        rf4mst['Wind Speed(4M)']=[data['Wind Speed'][119]]
        rf4mst['Energy_70m(MW)']=[rf4m['Energy_70m(MW)'][0]]
        rf4mst['10Turbines(MW)']=[rf4m['No_of_Turbines10TEnergy_70m(MW)'][0]]
        return render_template('Predict.html',tables=[rf4mst.to_html(classes='data')], titles=df.columns.values)
    elif (time==4.0 and radius==70.0 and turbines==15.0):
        rf4msf=pd.DataFrame()
        rf4msf['Air Density']=[data['Air Density'][119]]
        rf4msf['Wind Speed(4M)']=[data['Wind Speed'][119]]
        rf4msf['Energy_70m(MW)']=[rf4m['Energy_70m(MW)'][0]]
        rf4msf['15Turbines(MW)']=[rf4m['No_of_Turbines15TEnergy_70m(MW)'][0]]
        return render_template('Predict.html',tables=[rf4msf.to_html(classes='data')], titles=df.columns.values)
    elif (time==4.0 and radius==70.0 and turbines==20.0):
        rf4mstw=pd.DataFrame()
        rf4mstw['Air Density']=[data['Air Density'][119]]
        rf4mstw['Wind Speed(4M)']=[data['Wind Speed'][119]]
        rf4mstw['Energy_70m(MW)']=[rf4m['Energy_70m(MW)'][0]]
        rf4mstw['20Turbines(MW)']=[rf4m['No_of_Turbines20TEnergy_70m(MW)'][0]]
        return render_template('Predict.html',tables=[rf4mstw.to_html(classes='data')], titles=df.columns.values)
    elif (time==4.0 and radius==70.0 and turbines==30.0):
        rf4msth=pd.DataFrame()
        rf4msth['Air Density']=[data['Air Density'][119]]
        rf4msth['Wind Speed(4M)']=[data['Wind Speed'][119]]
        rf4msth['Energy_70m(MW)']=[rf4m['Energy_70m(MW)'][0]]
        rf4msth['30Turbines(MW)']=[rf4m['No_of_Turbines30TEnergy_70m(MW)'][0]]
        return render_template('Predict.html',tables=[rf4msth.to_html(classes='data')], titles=df.columns.values)
    elif (time==4.0 and radius==80.0 and turbines==10.0):
        rf4met=pd.DataFrame()
        rf4met['Air Density']=[data['Air Density'][119]]
        rf4met['Wind Speed(4M)']=[data['Wind Speed'][119]]
        rf4met['Energy_70m(MW)']=[rf4m['Energy_70m(MW)'][0]]
        rf4met['10Turbines(MW)']=[rf4m['No_of_Turbines10TEnergy_70m(MW)'][0]]
        return render_template('Predict.html',tables=[rf4met.to_html(classes='data')], titles=df.columns.values)
    elif (time==4.0 and radius==80.0 and turbines==15.0):
        rf4mef=pd.DataFrame()
        rf4mef['Air Density']=[data['Air Density'][119]]
        rf4mef['Wind Speed(4M)']=[data['Wind Speed'][119]]
        rf4mef['Energy_70m(MW)']=[rf4m['Energy_70m(MW)'][0]]
        rf4mef['15Turbines(MW)']=[rf4m['No_of_Turbines15TEnergy_70m(MW)'][0]]
        return render_template('Predict.html',tables=[rf4mef.to_html(classes='data')], titles=df.columns.values)
    elif (time==4.0 and radius==80.0 and turbines==20.0):
        rf4metw=pd.DataFrame()
        rf4metw['Air Density']=[data['Air Density'][119]]
        rf4metw['Wind Speed(4M)']=[data['Wind Speed'][119]]
        rf4metw['Energy_70m(MW)']=[rf4m['Energy_70m(MW)'][0]]
        rf4metw['20Turbines(MW)']=[rf4m['No_of_Turbines20TEnergy_70m(MW)'][0]]
        return render_template('Predict.html',tables=[rf4metw.to_html(classes='data')], titles=df.columns.values)
    elif (time==4.0 and radius==80.0 and turbines==30.0):
        rf4meth=pd.DataFrame()
        rf4meth['Air Density']=[data['Air Density'][119]]
        rf4meth['Wind Speed(4M)']=[data['Wind Speed'][119]]
        rf4meth['Energy_70m(MW)']=[rf4m['Energy_70m(MW)'][0]]
        rf4meth['30Turbines(MW)']=[rf4m['No_of_Turbines30TEnergy_70m(MW)'][0]]
        return render_template('Predict.html',tables=[rf4meth.to_html(classes='data')], titles=df.columns.values)
    elif (time==4.0 and radius==100.0 and turbines==10.0):
        rf4mht=pd.DataFrame()
        rf4mht['Air Density']=[data['Air Density'][119]]
        rf4mht['Wind Speed(4M)']=[data['Wind Speed'][119]]
        rf4mht['Energy_100m(MW)']=[rf4m['Energy_100m(MW)'][0]]
        rf4mht['10Turbines(MW)']=[rf4m['No_of_Turbines10TEnergy_100m(MW)'][0]]
        return render_template('Predict.html',tables=[rf4mht.to_html(classes='data')], titles=df.columns.values)
    elif (time==4.0 and radius==100.0 and turbines==15.0):
        rf4mhf=pd.DataFrame()
        rf4mhf['Air Density']=[data['Air Density'][119]]
        rf4mhf['Wind Speed(4M)']=[data['Wind Speed'][119]]
        rf4mhf['Energy_100m(MW)']=[rf4m['Energy_100m(MW)'][0]]
        rf4mhf['15Turbines(MW)']=[rf4m['No_of_Turbines15TEnergy_100m(MW)'][0]]
        return render_template('Predict.html',tables=[rf4mhf.to_html(classes='data')], titles=df.columns.values)
    elif (time==4.0 and radius==100.0 and turbines==20.0):
        rf4mhtw=pd.DataFrame()
        rf4mhtw['Air Density']=[data['Air Density'][119]]
        rf4mhtw['Wind Speed(4M)']=[data['Wind Speed'][119]]
        rf4mhtw['Energy_100m(MW)']=[rf4m['Energy_100m(MW)'][0]]
        rf4mhtw['20Turbines(MW)']=[rf4m['No_of_Turbines20TEnergy_100m(MW)'][0]]
        return render_template('Predict.html',tables=[rf4mhtw.to_html(classes='data')], titles=df.columns.values)
    elif (time==4.0 and radius==100.0 and turbines==30.0):
        rf4mhth=pd.DataFrame()
        rf4mhth['Air Density']=[data['Air Density'][119]]
        rf4mhth['Wind Speed(4M)']=[data['Wind Speed'][119]]
        rf4mhth['Energy_100m(MW)']=[rf4m['Energy_100m(MW)'][0]]
        rf4mhth['30Turbines(MW)']=[rf4m['No_of_Turbines30TEnergy_100m(MW)'][0]]
        return render_template('Predict.html',tables=[rf4mhth.to_html(classes='data')], titles=df.columns.values)        
if __name__ == '__main__':
    app.run(host='0.0.0.0',port=port,debug=False)
    