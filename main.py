import pandas as pd
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


def time_to_int(data):
    # Convert the 'date' column to datetime
    data['date'] = pd.to_datetime(data['date'])

    # Create additional features from the date
    data['hour'] = data['date'].dt.hour
    data['day'] = data['date'].dt.day
    data['month'] = data['date'].dt.month
    data['year'] = data['date'].dt.year
    data = data.drop(columns=['date'])
    return data
    


# Load the data from the CSV files into DataFrames and 
# convert the 'date' column to datetime format and create additional 
# features from the date column for both the training and test data sets. 
# remove all the NaN values from the 'SalePrice' column in the training data set.
X_train = time_to_int(pd.read_csv('1_X_train.csv'))
Y_train = time_to_int(pd.read_csv('1_y_train.csv'))   
X_test = time_to_int(pd.read_csv('1_X_test.csv'))
Y_test = time_to_int(pd.read_csv('1_y_train.csv')) 
Y_train = Y_train.iloc[9504:]
x_train,x_valid,y_train,y_valid=train_test_split(X_train,Y_train,train_size=0.8,test_size=0.2)


X_train = X_train.iloc[1133:]
model=RandomForestRegressor()
model.fit(X_train,y_train)

preds_valid=model.predict(x_valid)
score_valid=mean_absolute_error(y_valid,preds_valid)
print("MAE: ",score_valid)

preds_test=model.predict(X_test)
submission = pd.DataFrame({'location_id': Y_test.location_id,'hour': Y_test.hour, 'day': Y_test.day,'month': Y_test.month,'year': Y_test.year,'temperature_2m': Y_test.temperature_2m, 'relative_humidity_2m': Y_test.relative_humidity_2m })
submission.to_csv('1_sample_submission.csv',index=False)
    
