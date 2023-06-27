import yfinance as yf
from datetime import datetime

class Forex:
    """
    This gets the 5 minute resampled closing of USD_PHP forex data
    """
    def __init__(self,
                 start=datetime(2022, 9, 11),
                 end=datetime(2022, 11, 29, 23, 59)):
        self.start = start
        self.end = end
        self.interval = "60m"
    def download_forex(self):
        forex_df = yf.download("USDPHP=X",
                               start=self.start,
                               end=self.end,
                               interval=self.interval)
        forex_df.rename(columns={"Open": "USDPHP_Open",
                                 "High": "USDPHP_High",
                                 "Low": "USDPHP_Low",
                                 "Close": "USDPHP_Close",
                                 "Adj Close": "USDPHP_Adj_Close",
                                 "Volume": "USDPHP_Volume", },
                        inplace=True)
        forex_data = forex_df.resample("300S").pad()
        return forex_data