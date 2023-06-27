from datetime import datetime
import matplotlib.pyplot as plt
import meteostat as mtst
import pandas as pd
import os
import plotly.express as px
import psycopg2 as pg2

#For environment variables
class DatabaseManager():
    def __init__(self):
        self.user = os.environ["user"]
        self.password = os.environ["password"]
        self.dbname = os.environ["dbname"]
        self.host = os.environ["host"]
        self.db_url = f"postgresql+psycopg2://{self.user}:{self.password}@{self.host}:5432/{self.dbname}"
        self.table_name = "RTD_PRICE_SCHED_SEPT_NOV22"
        self.conn = pg2.connect(dbname=self.dbname,
                                user=self.user,
                                host=self.host,
                                password=self.password,)
        self.iemop_df = None
        self.selected_node = None
        self.iemop_df_plot = None
        self.iemop_df_roll_plot = None

    def make_query_and_process(self, node):
        """
        This gets the LMP data from the node from the local database
        :param node: the parameter node
        :return: LMP data plot and dataframe, this is unprocessed and uncleaned
        """
        self.selected_node = node
        df = pd.read_sql(sql=f"SELECT * FROM {self.table_name} WHERE \"RESOURCE_NAME\" ILIKE '%{self.selected_node}%'",
                         con=self.conn)
        self.iemop_df = df.drop(columns="Unnamed: 12", axis=1)
        self.iemop_df.RUN_TIME = pd.to_datetime(self.iemop_df.RUN_TIME)
        self.iemop_df.set_index("RUN_TIME", inplace=True)
        self.iemop_df.sort_index(ascending=True, inplace=True)

    def plot(self):
        self.iemop_df_plot = px.line(self.iemop_df,
                                     x=self.iemop_df.index,
                                     y="LMP",
                                     color="RESOURCE_NAME",
                                     title=f"LMP in {self.selected_node}",)
        return self.iemop_df_plot.to_html(), self.iemop_df