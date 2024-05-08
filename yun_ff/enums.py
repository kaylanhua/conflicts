import pandas as pd

supported_countries = [
    'colombia',
    'drc',
    'el_salvador',
    'ethiopia',
    'sierra_leone',
    'somalia',
    'south_sudan',
    'sri_lanka',
    'ukraine',
    'venezuela'
]

def add_date_range(country, tuples):
    WAR_DATES[country] = [(pd.to_datetime(start), pd.to_datetime(end)) for start, end in tuples]

WAR_DATES = {country: [] for country in supported_countries}
add_date_range('sri_lanka', [
    ('23 July 1983', '29 July 1987'), 
    ('June 10, 1990', 'January 1995'), 
    ('19 April 1995','22 February 2002'), 
    ('26 July 2006','18 May 2009')
])
add_date_range('colombia', [ # running since 1964
])
add_date_range('drc', [
    ('24 October 1996', '16 May 1997'), # first congo war
    ('2 August 1998', '18 July 2003'), # second congo war
    # ('June 5 1997', 'December 29 1999'),
])
add_date_range('somalia', [ # disputed dates
])