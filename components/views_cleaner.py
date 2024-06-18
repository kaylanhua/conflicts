## GENERAL IMPORTS 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

## CLASS DEF
class VIEWSCleaner():
    def __init__(self, filename, gw_id):
        self.features = self.load_data(filename, gw_id)
        self.features_war_dates = None
        self.gw_id = gw_id

    def load_data(self, filename, gw_id):
        all_features = pd.read_parquet(filename, engine='pyarrow')
        country_af = all_features[all_features["gleditsch_ward"] == gw_id]
        columns_to_remove = [col for col in all_features.columns if "_sb" in col or "ged" in col or "acled" in col][1:]
        return country_af.drop(columns=columns_to_remove)
    
    def plot(self):
        data = self.features

        dates = data["month_id"]
        target = data["ged_sb"]

        plt.figure(figsize=(12, 6))
        plt.plot(dates, target, label='UCDP Estimate')  # Changed plot to scatterplot
        plt.xlabel('Date')
        plt.ylabel('Fatalities')
        plt.gca().set_facecolor('#F9F5F1')
        plt.gca().spines['top'].set_color('#F9F5F1')
        plt.gca().spines['right'].set_color('#F9F5F1')
        plt.gcf().set_facecolor('#F9F5F1')

        plt.title(f'Fatalities Over Time (views {self.gw_id})')
        plt.legend(['Ground Truth'])
        plt.show()
        
    def set_war_var(self, data, date_ranges):
        # data = self.features
        
        def calculate_month_id(date):
            return 121 + (date.year - 1990) * 12 + (date.month - 1)

        int_dates = [(calculate_month_id(date), calculate_month_id(end)) for date, end in date_ranges]
        self.war_dates = int_dates
        print(self.war_dates)
        
        def populate_war_vars(date):
            months_since_start = np.nan
            for start, end in int_dates:
                if start <= date:
                    months_since_start = date - start
                if start <= date <= end:
                    return 1, months_since_start
            return 0, months_since_start

        # Apply the function to the 'date_start' column in the data
        data['wartime'], data['since_war_start'] = zip(*data['month_id'].apply(populate_war_vars))
        
        self.features_war_dates = data
        return data