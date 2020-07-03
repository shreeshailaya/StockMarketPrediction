import base64
import io
import json
import urllib
import requests
import PIL
import PIL.Image
import pandas as pd
import pandas_datareader as web
from django.shortcuts import HttpResponse
from django.shortcuts import render
from keras.layers import Dense, LSTM
from keras.models import Sequential
from matplotlib import pylab
from pylab import *
from sklearn.preprocessing import MinMaxScaler
from .forms import Portfolio
from textblob import TextBlob
from bs4 import BeautifulSoup
from django.contrib.auth.models import auth

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
    # if request.user.is_authenticated:
    if request.method == 'POST':
        search = request.POST['search']
        st_dis_name = search
        search = search.rpartition('.')[0]

        request_p = search
        ex = '.NS'
        st_name = request_p + ex
        end_date = datetime.date.today() - datetime.timedelta(days=1)
        end_date = end_date.strftime('%Y-%m-%d')
        df = web.DataReader(st_name, data_source='yahoo', start='2018-01-01', end=end_date)
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
        # volume graph
        plt.figure(figsize=(16, 8))
        plt.title('Volume')
        # plt.bar(df['Volume'])
        df['Volume'].plot(kind='bar')
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Volume in Cr', fontsize=18)
        buffer = io.BytesIO()
        canvas = pylab.get_current_fig_manager().canvas
        canvas.draw()
        pil_image = PIL.Image.frombytes("RGB", canvas.get_width_height(), canvas.tostring_rgb())
        pil_image.save(buffer, "PNG")
        buffer.seek(0)
        string = base64.b64encode(buffer.read())
        uri1 = 'data:image/png;base64,' + urllib.parse.quote(string)
        pylab.close()

        # ANN---------------------------------------------------------------------------------------------------

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

        # ANN Prediction Plot-----------------------------------------------------------------------------------

        # Plot the data
        train = data[:training_data_len]
        valid = data[training_data_len:]
        valid['Predictions'] = predictions
        # Visualize the data
        plt.figure(figsize=(16, 8))
        stock = 'NSE:' + request_p
        plt.title('ANN Prediction ')
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

        if pred_price[0][0] < df['Close'][len(df) - 1]:
            pred_feedback = 'Sell'
        elif pred_price[0][0] > df['Close'][len(df) - 1]:
            pred_feedback = 'Buy'
        else:
            pred_feedback = 'Hold'

        # SMA Chart Logic -----------------------------------------------------------------------------

        def sma_chart():
            # create 30 days Moving Avarge
            SMA30 = pd.DataFrame()
            SMA30['Adj Close price'] = df['Adj Close'].rolling(window=30).mean()
            SMA200 = pd.DataFrame()
            SMA200['Adj Close price'] = df['Adj Close'].rolling(window=100).mean()
            data_a = pd.DataFrame()
            data_a['Adj Close'] = df['Adj Close']
            data_a['SMA30'] = SMA30['Adj Close price']
            data_a['SMA200'] = SMA200['Adj Close price']
            p = len(data_a)

            if data_a['SMA200'][p - 1] > df['Close'][len(df) - 1]:
                sma_feedback = "Sell"
            elif data_a['SMA200'][p - 1] < df['Close'][len(df) - 1]:
                sma_feedback = "Buy"
            else:
                sma_feedback = "Hold"

            def buy_sell(data_a):
                sigPriceBuy = []
                sigPriceSell = []
                flag = -1

                for i in range(len(data_a)):
                    if data_a['SMA30'][i] < data_a['SMA200'][i]:
                        if flag != 1:
                            sigPriceBuy.append(data_a['Adj Close'][i])
                            sigPriceSell.append(np.nan)
                            flag = 1
                        else:
                            sigPriceBuy.append(np.nan)
                            sigPriceSell.append(np.nan)
                    elif data_a['SMA30'][i] > data_a['SMA200'][i]:
                        if flag != 0:
                            sigPriceBuy.append(np.nan)
                            sigPriceSell.append(data_a['Adj Close'][i])
                            flag = 0
                        else:
                            sigPriceBuy.append(np.nan)
                            sigPriceSell.append(np.nan)
                    else:
                        sigPriceBuy.append(np.nan)
                        sigPriceSell.append(np.nan)

                return sigPriceBuy, sigPriceSell

            # store Buy Sell Data in Variable
            buy_sell = buy_sell(data_a)
            data_a['Buy_Signal_Price'] = buy_sell[0]
            data_a['Sell_Signal_Price'] = buy_sell[1]
            # plot
            plt.figure(figsize=(16, 8))
            plt.plot(data_a['Adj Close'], label='Adj Close', alpha=0.25)
            plt.plot(SMA30['Adj Close price'], label='SMA30', alpha=0.25)
            plt.plot(SMA200['Adj Close price'], label='SMA200', alpha=0.50)
            plt.scatter(data_a.index, data_a['Sell_Signal_Price'], label='Sell', marker='v', color='red')
            plt.scatter(data_a.index, data_a['Buy_Signal_Price'], label='Buy', marker='^', color='green')
            plt.title('SMA Analysis')
            plt.xlabel('date')
            plt.ylabel('SMA Analysis')
            plt.legend(loc='upper left')
            buffer_a = io.BytesIO()
            canvas_a = pylab.get_current_fig_manager().canvas
            canvas_a.draw()
            pil_image_a = PIL.Image.frombytes("RGB", canvas_a.get_width_height(), canvas_a.tostring_rgb())
            pil_image_a.save(buffer_a, "PNG")
            buffer_a.seek(0)
            string_a = base64.b64encode(buffer_a.read())
            sma_ch = 'data:image/png;base64,' + urllib.parse.quote(string_a)
            pylab.close()
            return sma_ch, sma_feedback

        sma = sma_chart()

        # News---------------------------------------------------------------------------------------------------------
        def news_fetch():
            def percentage(part, whole):
                return 100 * float(part) / float(whole)

            url = 'https://www.bing.com/news/search?q=' + search + '&qs=n&form=NWRFSH'

            headers = {
                "User-Agent": 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:68.0) Gecko/20100101 Firefox/68.0'
            }
            page = requests.get(url, headers)
            soup = BeautifulSoup(page.content, 'html.parser')
            noOfSearch = 10
            positive = 0
            negative = 0
            neutral = 0
            polarity = 0
            news_dir = {}

            for x in range(noOfSearch):
                title = soup.findAll('a', {'class': 'title'})[x]
                analysis = TextBlob(title.text)
                source = soup.findAll('div', {'class': 'source'})[x]
                news_dir[title.text] = source.text
                news_url = soup.findAll('a', href=True)[x]

                polarity += analysis.sentiment.polarity

                if analysis.sentiment.polarity == 0.000:
                    neutral += 1

                elif analysis.sentiment.polarity < 0.000:
                    negative += 1

                elif analysis.sentiment.polarity > 0.000:
                    positive += 1

            positive = percentage(positive, noOfSearch)
            negative = percentage(negative, noOfSearch)
            neutral = percentage(neutral, noOfSearch)

            if polarity == 0:
                news_feedback = 'Hold'
            elif polarity < 0.000:
                news_feedback = 'Sell'
            elif polarity > 0.000:
                news_feedback = 'Buy'

            labels = ['Positive [' + str(positive) + '%]', 'Negative [' + str(negative) + '%]',
                      'Neutral [' + str(neutral) + '%]']
            sizes = [positive, negative, neutral]
            colors = ['lightblue', 'red', '#ffc107c2']
            patches, texts = plt.pie(sizes, colors=colors, startangle=90)
            plt.legend(patches, labels, loc="best", prop={'size': 6})
            plt.title('Latest News Analysis ')
            plt.axis('equal')
            plt.tight_layout()
            buffer_a = io.BytesIO()
            canvas_a = pylab.get_current_fig_manager().canvas
            canvas_a.draw()
            pil_image_a = PIL.Image.frombytes("RGB", canvas_a.get_width_height(), canvas_a.tostring_rgb())
            pil_image_a.save(buffer_a, "PNG")
            buffer_a.seek(0)
            string_a = base64.b64encode(buffer_a.read())
            news_feed = 'data:image/png;base64,' + urllib.parse.quote(string_a)
            pylab.close()

            return news_feed, news_dir, news_feedback, news_url

        news = news_fetch()
        # json-----------------------------------------------------------------------------------------------------------------------------------------------------
        api_request = requests.get(
            "https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=BSE:" + search + "&apikey=BP39EYKNTFWF2FOF")
        try:
            api = json.loads(api_request.content)
            st_name = api["Global Quote"]
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
                      'uri1': uri1,
                      'sma_ch': sma[0],
                      'pia_chart': news[0],
                      'news': news[1],
                      'news_feedback': news[2],
                      'pred_feedback': pred_feedback,
                      'sma_feedback': sma[1],
                      'dis_name': st_dis_name,
                      'news_url': news[3]
                      }

        return render(request, 'index.html', stock_info)
    else:
        return render(request, 'index.html', {'note': 'Please Enter Name'})


# else:
# return render(request, 'about.html')


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


def sCheck(request):
    return render(request, 'sCheck.html')


'''
        if pred_price[0][0] < st_name['05. price']:
            pred_feedback = 'Sell'
        elif pred_price[0][0] > st_name['05. price']:
            pred_feedback = 'Buy'
        else:
            pred_feedback = 'Hold'
            
            
            
            
                    if data_a['SMA200'][p - 1] > st_name['05. price']:
                sma_feedback = 'Sell'
            elif data_a['SMA200'][p - 1] < st_name['05. price']:
                sma_feedback = 'Buy'
            else:
                sma_feedback = 'Hold'
'''
