## GENERAL IMPORTS 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

## CLASS DEF
class VIEWSCleaner():
    def __init__(self, filename, gw_id, trim_full=True):
        self.features = self.load_data(filename, gw_id, trim_full)
        self.features_war_dates = None
        self.gw_id = gw_id

    def load_data(self, filename, gw_id, trim_full=True):
        all_features = pd.read_parquet(filename, engine='pyarrow')
        country_af = all_features[all_features["gleditsch_ward"] == gw_id]
        target_vars = ["ged_os", "ged_ns", "acled_sb", "acled_sb_count", "acled_os"]
        full_columns = [col for col in all_features.columns if "_sb" in col or "ged" in col or "acled" in col][1:]
        if trim_full:
            columns_to_remove = full_columns
        else:
            columns_to_remove = target_vars
            
        print(f"Removing columns: {columns_to_remove}")
        return country_af.drop(columns=columns_to_remove)
    
    def plot(self, n=None):
        data = self.features

        dates = data["month_id"]
        target = data["ged_sb"]

        plt.figure(figsize=(12, 6))
        plt.plot(dates, target, label='UCDP Estimate', alpha=0.7)
        
        if n is not None:
            rolling_avg = target.rolling(window=n).mean()
            plt.plot(dates, rolling_avg, label=f'{n}-Month Rolling Average', color='red', linewidth=2)
        
        plt.xlabel('Date')
        plt.ylabel('Fatalities')
        plt.gca().set_facecolor('#F9F5F1')
        plt.gca().spines['top'].set_color('#F9F5F1')
        plt.gca().spines['right'].set_color('#F9F5F1')
        plt.gcf().set_facecolor('#F9F5F1')

        plt.title(f'Fatalities Over Time (views {self.gw_id})')
        plt.legend()
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