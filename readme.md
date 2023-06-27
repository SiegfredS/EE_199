# This is the general idea for my EE 199 Project

The project is titled "A Web-Based Forecasting Tool for 
Hourly Rolling Average Locational Marginal Prices Through
Supervised Machine Learning"

This project gets user inputs on the location, node name,
as well as machine learning model name. The project then 
gets the longitude and latitude through geopy, and then 
subsequently gets the weather parameter through meteostat.
Forex rates are also gathered through yfinance. The 
locational marginal price of the node is also loaded
through a local database. These are then plotted for the
user to verify. After which, the data is processed and ML
models are fitted. The error metrics of these models, their
runtime, feature importance and kde-plots are shown. The
time-series plot of the selected model, as well as its
feature importance are also shown to the user.

The explanations of the different files are shown below.
For more information on the project, please refer to the
project manuscript.

## cleaner.py

This file cleans the static folder which contains the 
pictures of the graphs shown every time the code is run.
This is to show only the necessary plots asked by the user.

## dbmanager.py

This manages the database which contains the data from
the Independent Electricity Market Operator from Sept. 2022
to Nov. 2022.

## forms.py

This contains the prompts to the users rendered into html
in the user interface

## machinelearner.py

This contains the code for the machine learning models
used in the project. It also has modules to fit the model,
to output the timeseries forecast, to output error metrics,
to output feature importance, and other relevant ML
parameters

## main.py

This contains the Flask app, and acts as the server of this
project.

## preprocessor.py

This contains the modules for processing the data. The
processing contains winsorization, and yeo-johnson
transformation as well as some feature engineering
techniques in order to extract more information from the
data.

## usdphp.py

This extracts the forex rates using yfinance package
for the relevant dates.

## weatherlocator.py

This locates the longitude and latitude of the
selected locations as well as gets the weather parameters
from a package called meteostat.