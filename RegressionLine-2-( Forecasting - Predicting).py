
import pandas as pd
import quandl
import math, datetime
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

## (WE HAVE MADE A FEW CHANGES HERE ) ------>>>>>..................................

import matplotlib.pyplot as plt
from matplotlib import style

style.use("ggplot")
## To plt a Graph for the Forecast



## (TILL HERE) ------>>>>>..................................


## PREPROCESSING->using this for scalling the data of the Features, the value of features should be somewhere between  -1 and +1, HELPSSING Speed
## sklearn.model_selection import train_test_split  -> To Create and Training esting Samples, Nice Way to split up data and shuffle it which helps in statistics to not have a bioased sample

## svm (support Vector Machine)

##NUMPY -> A Computing library helps to use ARRAY- since python does not have ARRAYS

df=quandl.get('WIKI/GOOGL')

print(df.columns) ## This prints all the columns of the data frame

df=df[ ['Adj. Open','Adj. High', 'Adj. Low', 'Adj. Close','Adj. Volume' ] ]
df['HL_PCT']=(df['Adj. High'] -df['Adj. Close'] )/ df['Adj. Close']*100.0
df['PCT_change']=(df['Adj. Close'] -df['Adj. Open'] )/ df['Adj. Open']*100.0

df = df[ ['Adj. Close','HL_PCT','PCT_change','Adj. Volume']  ] ## These columns are the  features i.e. the attributes


##-----------------------------------------------------------------------------

forecast_col='Adj. Close' ##Variable for a forecast coulumn to use linear regression on

df.fillna(-99999,inplace=True)  ## To fil NULL/ NAN type values, because in machine learning we can't work with NAN values
## We add an oultier value i.e. -99999 in this case to replace NaN values

forecast_out=int(math.ceil(0.01*len(df)))   ## Using Linear  Regression to forecast values
print(forecast_out)
## int because math.ceil() - returns float not int
## math.ceil-> It rounds the number to the nearest Integer
## 0.01 -> the percentage of Data Frame we want to use
##No of days in advance


df['label']=df[forecast_col].shift(-forecast_out) ## This will adjust the Adj.Close price some days into the future i.e. 0.01% days into the future


## DEFINED FEARURES = X  (capital)
## DEFINED LABELS = y   (small)


X = np.array(df.drop(['label'],1))  ## Features are everything except the label column
## df.drop() -> It is returning the new dataframe


## (WE HAVE MADE A FEW CHANGES HERE ) ------>>>>>..................................


''' SCALLING X '''

X = preprocessing.scale(X)
## We are scalling X, before we feed it to the classifier,we include it with training data.
## When we have a classifier we are using it real time on real data
## When we go to the future we need to scale the values alongside all the other  values, this can add to the prodessing time while training and testing

X_lately = X[-forecast_out:]
X = X[:-forecast_out]
## X_lately -> It is the stuff we are going to predict against
## We have the X now we find out M and B for (y = MX + B)
## X_lately -> It is data we dont have a y-Value

## (TILL HERE) ------>>>>>..................................


df.dropna(inplace=True)
y = np.array(df['label'])
y = np.array(df['label'])

##-----------------------------------------------------------------------------


'''NOW CREATING TRAINING & TESTING SETS'''


X_train,X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

## TEST_SIZE ->> We can choose how much data we mant to use to test and train
## cross_validate.train_test_split(X,y, test_size=0.2) ->> It is taking all the features and labels, shuffles them up keeping X and y connected, and output the Testing and Training data for X and y

clf = LinearRegression()
 
## For X and y TRAIN we fit our Classifiers, clf-> Is a classifier

clf.fit(X_train,y_train) ## FIT for TESTING

accuracy = clf.score(X_test,y_test) ## SCORE for TRAINING


## We TEST and TRAIN on separate data, so that during testing after Training we will be asking the exactly the  same parameters

print(accuracy)


## ACCURACY -> It is SQUARED ERROR

##-----------------------------------------------------------------------------


'''PREDICTING DATA'''

## We need  X Data, We do it USING CLASSIFIER

forecast_set = clf.predict(X_lately) ## X_lately is values of X for past 0.01% days ---->>> THE MAIN THING THAT PREDICTS THE VALUES

print(forecast_set,"\n", accuracy, "\n",forecast_out)

df['Forecast']=np.nan

last_date=df.iloc[-1].name ## Very Last Date
last_unix = last_date.timestamp()
one_day=86400 ## It is no of Seconds in a day
next_unix = last_unix + one_day ## Next_Unix is Next Day


''' POPULATING THE DATA FRAME WITH NEW DATES AND FORECAST VALUES '''

## THIS FOR LOOP IS TO PUT DATES ON THE AXIS
for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix) ## We are iterating through the Forecast set taking each day and setting those values in the data frame
    next_unix += one_day ## Updates the next_unix for the next Iteration
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]
    
    ## df.loc references the index,   df.loc[next_date]-->> Means Next_Date is the index of the data Frame
    ## [np.nan for _ in range(len(df.columns)-1)] -->> LIST of Values the are np.nan
    ## + [i] -->> Means that we are adding the Forecast in the end (another column)
    ## Takes all the first columns sets them to NaN and the final column is [i] i.e. forecast
    ## Basically the LAST LINE changes the structure of a DataFrame by inserting rows with blank values

print(df.tail())

df['Adj. Close'].plot() ## Known Data
df['Forecast'].plot() ## Predicted Data
plt.legend(loc=4) ## Legend is area describing he elements of the graph
plt.xlabel('Data')
plt.ylabel('Price')
plt.show()
 

