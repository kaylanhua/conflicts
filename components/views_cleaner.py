## GENERAL IMPORTS 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

## CLASS DEF
class VIEWSCleaner():
    def __init__(self, filename, country_id):
        self.features = self.load_data(filename, country_id)
        self.country_id = country_id

    def load_data(self, filename, country_id):
        all_features = pd.read_parquet(filename, engine='pyarrow')
        country_af = all_features[all_features["country_id"] == country_id]
        columns_to_remove = [col for col in all_features.columns if "_sb" in col or "ged" in col or "acled" in col][1:]
        return country_af.drop(columns=columns_to_remove)
    
    def plot(self, user_data=None, use_resampled=False):
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
        
        plt.title(f'Fatalities Over Time (views {self.country_id})')
        plt.legend()
        plt.show