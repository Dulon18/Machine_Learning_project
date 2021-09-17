import pickle
import streamlit as st
import numpy as np

st.title('Laptop Price Predictor')

#import model
pipe = pickle.load(open('pipe.pkl','rb'))
laptop_df = pickle.load(open('laptop_df.pkl','rb'))

#brand
company=st.selectbox('Brand',laptop_df['Company'].unique())

#type of laptop

type =st.selectbox('Type',laptop_df['TypeName'].unique())

#Ram
ram =st.selectbox('RAM( in GB)',[2,4,6,8,12,16,32,64])

#weight
weight = st.number_input("Weight of the laptop")

#TouchScreen

ts=st.selectbox('Touchsreen',['No','Yes'])

#ips
Ips=st.selectbox('IPS',['No','Yes'])

# screen size

screen_size= st.number_input('Screen Size')

#resolution

rs=st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800',
                                     '2880x1800','2560x1600','2560x1440','2304x1440'])

#Cpu,HDD,SSD,GPU brand,os

cpu=st.selectbox('CPU',laptop_df['CPU brand'].unique())
hdd=st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])
ssd=st.selectbox('SSD(in GB)',[0,8,128,512,1024])
gpu=st.selectbox('GPU',laptop_df['Gpu Brand'].unique())
os=st.selectbox('Operating System',laptop_df['OS'].unique())

#button
if st.button("Predict Price"):

    if ts=='Yes':
        ts=1
    else:
        ts=0

    if Ips=='Yes':
        ips=1
    else:
        Ips=0

    x_res = int(rs.split('x')[0])
    y_res = int(rs.split('x')[1])
    ppi   = ((x_res ** 2)  + (y_res ** 2)) ** (1/2) / screen_size
    query = np.array([company,type,ram,weight,ts,Ips,ppi,cpu,hdd,ssd,gpu,os])

    query=query.reshape(1,12)
    st.title(" The Laptop price will be  " + str(int(np.exp(pipe.predict(query)[0])) ) +"  Tk")
    
    
