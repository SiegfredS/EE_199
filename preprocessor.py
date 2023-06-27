import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer
from scipy.stats.mstats import winsorize
from datetime import datetime
from sklearn.model_selection import train_test_split

class PreProcessor:

    def __init__(self,
                 iemop_data: pd.DataFrame,
                 weather_data: pd.DataFrame,
                 forex_data: pd.DataFrame):
        self.iemop_data = iemop_data
        self.weather_data = weather_data
        self.forex_data = forex_data
        self.data = None
        self.shifted_data = None
        self.pt = None
        self.train_ratio = 0.5
        self.cross_ratio = 0.3
        self.test_ratio = 0.2
        self.X_train = self.X_test = self.X_cross = self.y_train = self.y_test = self.y_cross = None
        self.time_to_forecast = "1w"
        self.param_to_forecast = f"LMP_movave_1h_shifted_{self.time_to_forecast}_forward"
        self.merge()
        self.drop_specific_columns()
        self.get_rolling_temporal()
        self.process_data()
        self.shift_data()
        self.split_data()

    def merge(self):
        """
        merges all the current pandas dataframe
        :return: merged data frame in self.data
        """
        merged_df_1 = pd.merge(self.iemop_data,
                               self.weather_data,
                               left_index=True,
                               right_index=True)
        self.data = pd.merge(merged_df_1,
                             self.forex_data,
                             how="inner",
                             left_index=True,
                             right_index=True)

    def drop_specific_columns(self):
        """
        Drops columns that have no relevance to the project, especially those that do not bear any
        significance for the machine learning model.
        A few example are:
        snow - it doesn't snow in the Philippines
        tsun - while total sunshine is good, it is not measured by meteostat in the philippines
        Region_name - is the same for the same location
        :return:data with cleaned columns
        """
        columns_to_drop = ["snow",
                           "wpgt",
                           "tsun",
                           "USDPHP_Volume",
                           "SCHED_MW",
                           "MKT_TYPE",
                           "REGION_NAME",
                           "TIME_INTERVAL",
                           "RESOURCE_NAME",
                           "RESOURCE_TYPE",
                           "USDPHP_Open",
                           "USDPHP_High",
                           "USDPHP_Low",
                           "USDPHP_Adj_Close",]
        self.data.drop(columns=columns_to_drop, inplace=True)

    def get_rolling_temporal(self):

        """
        Gets moving averages and temporal parameters such as:
        quarter, month, week of the year, day of the week, hour, and the ix parameter
        the ix parameter is the sequential parameter of the data
        :return:
        """
        # Get moving averages

        to_roll = {"1h": 12,
                   "1d": 288,
                   "1w": 2016}

        for key, value in to_roll.items():
            self.data[[f"LMP_movave_{key}", f"LMP_movstd_{key}"]] = self.data.LMP.rolling(value).agg([np.mean, np.std])

        # temporal
        self.data["minute"] = self.data.index.minute
        self.data["hour"] = self.data.index.hour
        self.data["qtr"] = self.data.index.quarter
        self.data["mon"] = self.data.index.month
        self.data["week"] = self.data.index.week
        self.data["day"] = self.data.index.weekday

        # give another variable "ix" to count the sequence of the data aside from the run_time index
        self.data["seq_num"] = range(0, len(self.data))
        # drop not a number
        self.data.dropna()

    def process_data(self):
        """

        :return:
        """
        to_transform = ['LMP', 'LOSS_FACTOR', 'LMP_SMP', 'LMP_LOSS', 'LMP_CONGESTION', 'temp',
                        'dwpt', 'rhum', 'prcp', 'wspd', 'pres', 'USDPHP_Close',
                        'LMP_movave_1h', 'LMP_movstd_1h', 'LMP_movave_1d', 'LMP_movstd_1d',
                        'LMP_movave_1w', 'LMP_movstd_1w']
        self.pt = PowerTransformer(method="yeo-johnson")

        for var in to_transform:
            self.data[var] = winsorize(self.data[var], limits=(0.01, 0.01))  # limits at 1st and 99th percentiles
            self.data[var] = self.pt.fit_transform(self.data[var].values.reshape(-1, 1))
        pass

    def shift_data(self):
        self.shifted_data = self.data.copy()
        column_to_shift = "LMP_movave_1h"
        shift_steps = {"1h": 12,
                       "1d": 288,
                       "1w": 2016}

        for key, value in shift_steps.items():
            self.shifted_data[f"{column_to_shift}_shifted_{key}_ago"] = self.shifted_data[column_to_shift].shift(value)
        self.shifted_data[f"{column_to_shift}_shifted_{self.time_to_forecast}_forward"] = self.shifted_data[column_to_shift].shift(-shift_steps[self.time_to_forecast])
    def split_data(self):
        forecast_df = self.shifted_data
        forecast_df = forecast_df.dropna()
        #Get train muna
        X_train, X_test, y_train, y_test = train_test_split(forecast_df.drop(columns=self.param_to_forecast),
                                                            forecast_df[self.param_to_forecast],
                                                            test_size=1-self.train_ratio,
                                                            random_state=42,)
        #Get cross and test
        X_cross, X_test, y_cross, y_test = train_test_split(X_test,
                                                            y_test,
                                                            test_size=self.test_ratio/(self.test_ratio + self.cross_ratio),
                                                            random_state=42)
        self.X_train = X_train
        self.X_test = X_test
        self.X_cross = X_cross
        self.y_train = y_train
        self.y_test = y_test
        self.y_cross = y_cross

    def retrieve_data(self):
        return self.X_train, self.X_cross, self.X_test, self.y_train, self.y_cross, self.y_test


