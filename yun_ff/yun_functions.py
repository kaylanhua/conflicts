## GENERAL IMPORTS 
from difflib import get_close_matches
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

## MY IMPORTS
import enums 

## FUNCTIONS
def fuzzy_match(input):
    """
    input: string given by user in search
    """
    closest_match = get_close_matches(input, enums.supported_countries, n=1, cutoff=0.6)
    return closest_match[0] if closest_match else None


## CLASSES
class UCDPCleaner():
    def __init__(self, filename, use_high=False, full_path=None):
        self.filename = filename
        self.data = self.load_data(use_high, full_path)
        self.region_name = filename.split('_')[0].capitalize()
        self.resampled = None
        self.data_war_vars = None
        self.war_dates = None
        
    def load_data(self, use_high, full_path):
        fn = '../data/ucdp/' + self.filename + '.csv'
        if full_path is not None:
            fn = full_path
        print("Loading data from", fn)
        
        try: ucdp = pd.read_csv(fn)
        except: 
            print(f'Could not read {fn}')
            return
        ucdp['date_start'] = pd.to_datetime(ucdp['date_start'])
        ucdp_sorted = ucdp.sort_values(by="date_start")
        
        ## use high estimates if best is zero 
        if use_high:
            ucdp_sorted.loc[ucdp_sorted['best'] == 0, 'best'] = ucdp_sorted.loc[ucdp_sorted['best'] == 0, 'high']
        
        return ucdp_sorted
    
    def plot(self, user_data=None, use_resampled=False):
        
        ucdp = self.data if not use_resampled else self.resampled
        if user_data is not None:
            ucdp = user_data
        # ucdp['date_start'] = pd.to_datetime(ucdp['date_start'])
        # ucdp_sorted = ucdp.sort_values(by="date_start")

        dates_ucdp = ucdp["date_start"] if not use_resampled else ucdp.index
        target_ucdp = ucdp["best"]

        print(f'dates range from {min(dates_ucdp).date()} to {max(dates_ucdp).date()}')
        plt.figure(figsize=(12, 6))
        plt.plot(dates_ucdp, target_ucdp, label='UCDP Estimate')  # Changed plot to scatterplot
        plt.xlabel('Date')
        plt.ylabel('Fatalities')
        plt.gca().set_facecolor('#F9F5F1')
        plt.gca().spines['top'].set_color('#F9F5F1')
        plt.gca().spines['right'].set_color('#F9F5F1')
        plt.gcf().set_facecolor('#F9F5F1')
        
        plt.title(f'Fatalities Over Time (UCDP {self.region_name})')
        plt.legend()
        plt.show
        
    def plot_war(self):
        if self.data_war_vars is None or self.resampled is None:
            print('Please run set_war_var and resample first')
            return
        data = self.resampled
        # Plotting the data with wartime periods shaded
        plt.figure(figsize=(10, 5))
        plt.plot(data.index, data['best'], label='Data')  
        
        ## TODO: combine this function w the one above
        for start, end in self.war_dates:
            if data[(data.index >= start) & (data.index <= end)].shape[0] > 0:
                plt.axvspan(start, end, color='red', alpha=0.3)
        plt.legend()
        plt.title(f'{self.region_name} Fatalities with Wartime Periods Shaded')
        plt.xlabel('Date')
        plt.ylabel('Fatalities') 
        plt.gca().set_facecolor('#F9F5F1')
        plt.gca().spines['top'].set_color('#F9F5F1')
        plt.gca().spines['right'].set_color('#F9F5F1')
        plt.gcf().set_facecolor('#F9F5F1')
        
        plt.show()
    
    def cut_off_end(self, date):
        date = datetime.strptime(date, '%m-%d-%Y')
        data = self.data
        data = data[data['date_start'] < date]
        return data 
    
    def set_war_var(self, date_ranges):
        # do before resampling
        data = self.data
        self.war_dates = date_ranges
        
        def in_wartime_and_days_since_start(date):
            days_since_start = np.nan
            for start, end in date_ranges:
                if start <= date:
                    days_since_start = (date - start).days
                if start <= date <= end:
                    return 1, days_since_start
            return 0, days_since_start

        # Apply the function to the 'date_start' column in the data
        data['wartime'], data['since_war_start'] = zip(*data['date_start'].apply(in_wartime_and_days_since_start))
        # print(data['wartime'].value_counts(normalize=True))
        self.data_war_vars = data
        return data

            
    def duration(self):
        data = self.data
        data['date_start'] = pd.to_datetime(data['date_start'])
        data['date_end'] = pd.to_datetime(data['date_end'])
        data['duration'] = data['date_end'] - data['date_start'] 
        
        return data
    
    def set_data(self, new_data):
        self.data = new_data
    
    def resample(self, time_length, war_var=False):
        data = self.data
        data_resampled = data.set_index('date_start').resample(time_length).agg({
            'best': 'sum', # sum of fatalities
            'conflict_new_id': lambda x: x.nunique(), # number of unique conflicts
            'duration': 'mean', # average duration of conflict
            'id': 'count', # number of events
            'wartime': lambda x: 1 if x.mean() > 0.5 else 0, # whether the month is in wartime based on majority vote
            'since_war_start': 'mean' # amount of time since the start of the war
        }).rename(columns={'conflict_new_id': 'unique_conflicts', 'duration': 'avg_duration', 'id': 'events_count'})
        
        self.resampled = data_resampled
        
        return data_resampled

