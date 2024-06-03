import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st

start = '2010-01-01'
end = '2019-12-31'


st.title('stock trend prediction')
user_input = st.text_input('Enter Stock Ticker', 'AAPL')
df = data.DataReader(user_input,'yahoo', start, end)



#Describing data
st.subheader('Data from 2010 - 2019')
st.write(df.describe())

#visualization
st.subheader('closing Price vs Time chart')
fig = plt.figure(figsize= (12,6))
plt.plot(df.Close, 'b')
st.pyplot(fig)

st.subheader('closing Price vs Time chart with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize= (12,6))
plt.plot(ma100, 'r')
plt.plot(df.Close, 'b')
st.pyplot(fig)

st.subheader('closing Price vs Time chart with 200 and 100MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize= (12,6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(df.Close, 'b')
st.pyplot(fig)


#splitting data into training and testing

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70) : int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)


#load model
model = load_model('keras_model.h5')

#testinf part
past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index=True)#connecting training and testing 
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test, y_test = np.array(x_test), np.array(y_test)

#making predictions

y_predicted = model.predict(x_test)
scaler=scaler.scale_

scale_factor = 1/scaler[0]
y_predicted=y_predicted*scale_factor
y_test=y_test*scale_factor

#final graph
st.subheader('predictions vs original')
fig2=plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label='ori')
plt.plot(y_predicted, 'g', label='predicted' )
plt.xlabel('time')
plt.ylabel('price')
plt.legend()
st.pyplot(fig2)