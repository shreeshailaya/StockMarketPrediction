import base64
import io
import json
import urllib
import PIL
import PIL.Image
import pandas_datareader as web
import requests
from django.shortcuts import HttpResponse
from django.shortcuts import render
from keras.layers import Dense, LSTM
from keras.models import Sequential
from matplotlib import pylab
from pylab import *
from sklearn.preprocessing import MinMaxScaler
from .forms import Portfolio

plt.style.use('fivethirtyeight')
'''
import math
from django.contrib import messages
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.backends.backend_svg import FigureCanvas
from matplotlib.figure import Figure
from datetime import datetime
import tensorflow as tf
'''


def about(request):
    return render(request, 'about.html')


def index(request):
    if request.method == 'POST':
        search = request.POST['search']

        request_p = search
        ex = '.NS'
        st_name = request_p + ex
        end_date = datetime.date.today() - datetime.timedelta(days=1)
        end_date = end_date.strftime('%Y-%m-%d')
        df = web.DataReader(st_name, data_source='yahoo', start='2019-01-01', end=end_date)
        print(df)
        plt.figure(figsize=(16, 8))
        plt.title('Close Price History')
        plt.plot(df['Close'])
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Close Price in RS', fontsize=18)
        buffer = io.BytesIO()
        canvas = pylab.get_current_fig_manager().canvas
        canvas.draw()
        pil_image = PIL.Image.frombytes("RGB", canvas.get_width_height(), canvas.tostring_rgb())
        pil_image.save(buffer, "PNG")
        buffer.seek(0)
        string = base64.b64encode(buffer.read())
        uri = 'data:image/png;base64,' + urllib.parse.quote(string)
        pylab.close()

        # create data frame with only the close
        data = df.filter(['Close'])
        # convert dataframe to numpy array
        dataset = data.values
        # Get the number of rows to tain the model on 80%
        training_data_len = math.ceil(len(dataset) * .8)
        # scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)
        # CREATE TRAINING DATASET
        # CREATE SCALED TRAINING DATASET
        train_data = scaled_data[0:training_data_len, :]
        # SPLIT THE DATA INTO XTRAIN AND YTRAIN
        x_train = []
        y_train = []

        for i in range(60, len(train_data)):
            x_train.append(train_data[i - 60:i, 0])
            y_train.append(train_data[i, 0])
            if i <= 61:
                print(x_train)
                print(y_train)
        # CONVERT THE X_TRAIN AND Y_TRAIN TO NUMPY ARRAY
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        # RESHAPE THE DATA
        # 2D TO 3D
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        # BUILD LSTM MODEL
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))
        # COMPILE THE MODEL
        model.compile(optimizer='adam', loss='mean_squared_error')
        # TRAIN THE MODELS
        model.fit(x_train, y_train, batch_size=1, epochs=1)
        # CREATE TESTING DATASET
        # CREATE A ARRAY CONTAINING SCALED VALUES
        test_data = scaled_data[training_data_len - 60:, :]
        # CREATE THE DATASET X_TEST AND Y_TEST
        x_test = []
        y_test = dataset[training_data_len:, :]
        for i in range(60, len(test_data)):
            x_test.append(test_data[i - 60:i, 0])
        # CONVERT DATA INTO NUMPY ARRAY
        x_test = np.array(x_test)
        # RESHAPE THE DATE
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        # Get the models predicted price values
        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)
        # Get the root mean squared error (RMSE)
        rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))

        # Plot the data
        train = data[:training_data_len]
        valid = data[training_data_len:]
        valid['Predictions'] = predictions
        # Visualize the data
        plt.figure(figsize=(16, 8))
        stock = 'NSE:' + request_p
        plt.title(stock)
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Close Price USD ($)', fontsize=18)
        plt.plot(train['Close'])
        plt.plot(valid[['Close', 'Predictions']])
        plt.legend(['Train', 'Test', 'Predictions'], loc='upper left')
        buffer = io.BytesIO()
        canvas = pylab.get_current_fig_manager().canvas
        canvas.draw()
        pil_image = PIL.Image.frombytes("RGB", canvas.get_width_height(), canvas.tostring_rgb())
        pil_image.save(buffer, "PNG")
        buffer.seek(0)
        string = base64.b64encode(buffer.read())
        uri_pre = 'data:image/png;base64,' + urllib.parse.quote(string)
        pylab.close()

        # Get the quote
        quote = web.DataReader(st_name, data_source='yahoo', start='2019-01-01', end=end_date)
        # Create a new dataframe
        new_df = quote.filter(['Close'])
        # Get teh last 60 day closing price values and convert the dataframe to an array
        last_60_days = new_df[-60:].values
        # Scale the data to be values between 0 and 1
        last_60_days_scaled = scaler.transform(last_60_days)
        # Create an empty list
        X_test = [last_60_days_scaled]
        # Append teh past 60 days
        # Convert the X_test data set to a numpy array
        X_test = np.array(X_test)
        # Reshape the data
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        # Get the predicted scaled price
        pred_price = model.predict(X_test)
        # undo the scaling
        pred_price = scaler.inverse_transform(pred_price)

        api_request = requests.get(
            "https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=NSE:" + search + "&apikey=BP39EYKNTFWF2FOF")
        # api_request1 = requests.get("https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY_ADJUSTED&symbol=NSE:" + search + "&apikey=BP39EYKNTFWF2FOF")
        try:
            api = json.loads(api_request.content)
            st_name = api["Global Quote"]

            # api_ca = json.loads(api_request1.content)
            # api_ca2 = api_ca['Meta Data']
            # api_ca3 = api_ca2['announcements']
            # type(api_ca2)

        except Exception as e:
            st_name = "Error"

        stock_info = {'symbol': st_name['01. symbol'],
                      'open': st_name['02. open'],
                      'high': st_name['03. high'],
                      'low': st_name['04. low'],
                      'price': st_name['05. price'],
                      'volume': st_name['06. volume'],
                      'ltrday': st_name['07. latest trading day'],
                      'pclose': st_name['08. previous close'],
                      'change': st_name['09. change'],
                      'changep': st_name['10. change percent'],
                      'image': uri,
                      'image_pre': uri_pre,
                      'ca': pred_price,
                      }

        return render(request, 'index.html', stock_info)
    else:
        return render(request, 'index.html', {'note': 'Please Enter Name'})


def add(request):
    if request.method == 'POST':
        folio = Portfolio(request.POST or None)
        if folio.is_valid():
            folio.save()
            # messages.success(request, "stock has been added")
            return HttpResponse("Successfully Saved in DB")
        else:
            return HttpResponse("NOT Saved in DB")

    else:
        return render(request, 'add_stocks.html')
