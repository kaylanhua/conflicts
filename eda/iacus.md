# iacus notes
from 4.10 chat

- use spikes (set a threshold)
- keep the spikes in the data, include the probability that there is another spike 
- start doing aggregation at country level
- take the counts, average duration, and number of conflicts
- pred should be much better 
- time series, temporal focus 
- for south sudan, start with monthly

- if you have too many zeros, this will be an issue for any model 
- probability of zeros too
- given the data, what is the probability of a spike or zero 

- go by day , count casualties and num of conflicts and a 
  - use this to estimate periods of peace and periods of spikes
    
### main analysis use monthly data
- see each month (cumulative probability)
- conflict or not , probability or not by day
- cumulate by month and use that as a feature 
- month: total number of casualties, total number of conflicts, average duration of conflict, number of events in general 
- hf daily data: take the same measures and then we'll see 