#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error


# In[3]:


df = pd.read_csv('NTFLX.csv')
df


# In[4]:


df.head()


# In[5]:


# Step 3: Prepare the data for LSTM model
# Use only the 'Close' price for prediction
data = df[['Close']].values


# In[6]:


data


# In[7]:


# Normalize the data to values between 0 and 1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)


# In[8]:


# Split the data into training (80%) and testing (20%) sets
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]


# In[9]:


# Create sequences for LSTM input
def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


# In[10]:


sequence_length = 60  # We will use 60 days of data to predict the next day's stock price
X_train, y_train = create_sequences(train_data, sequence_length)
X_test, y_test = create_sequences(test_data, sequence_length)


# In[11]:


# Reshape the data for LSTM input (samples, time steps, features)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


# In[12]:


# Step 4: Build the LSTM model
model = Sequential()


# In[13]:


model


# In[14]:


# Add LSTM layers
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=False))


# In[15]:


# Add Dense layers
model.add(Dense(units=25))
model.add(Dense(units=1))


# In[16]:


# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')


# In[21]:


# Step 5: Train the model
model.fit(X_train, y_train, batch_size=64, epochs=10)


# In[22]:


# Step 6: Make predictions using the LSTM model
predictions = model.predict(X_test)


# In[23]:


# Inverse scaling to get the actual stock price
predictions = scaler.inverse_transform(predictions)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))


# In[24]:


# Step 7: Visualize the predictions and actual stock prices
plt.figure(figsize=(10, 6))
plt.plot(y_test_actual, color='blue', label='Actual Stock Price')
plt.plot(predictions, color='red', label='Predicted Stock Price')
plt.title('Stock Price Prediction (LSTM)')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()


# In[25]:


# Step 8: Evaluate the model using RMSE
rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))
print(f'Root Mean Squared Error (RMSE): {rmse}')

