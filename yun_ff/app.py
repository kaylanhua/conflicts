import streamlit as st
import streamlit.components.v1 as components
import yun_functions as yf
import enums 

# streamlit run app.py
st.set_page_config(layout="wide")
st.title('conflict analysis')
st.write('an interactive exploration of conflict data & modeling')

## ----------------------------------------------
## STYLING
## ----------------------------------------------
def styling(hex_color):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: {hex_color};
        }}
        div.stButton > button:first-child {{
            margin-top: 1.5em;
            height: 2.6em;   
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

styling('#F9F5F1')
st.set_option('deprecation.showPyplotGlobalUse', False)

## ----------------------------------------------
## CONSTANTS
## ----------------------------------------------
RAW_DATA = None

## ----------------------------------------------
## FUNCTIONS
## ----------------------------------------------
def open_viz(cleaner: yf.UCDPCleaner):
    # add duration information and show user overview
    cleaner.set_data(cleaner.duration())
    RAW_DATA = cleaner.data
    st.write(f'there were {RAW_DATA.shape[0]} entries found for violence in {country} between {RAW_DATA.date_start.min()} and {RAW_DATA.date_start.max()}')
    
    # plot the raw data
    st.pyplot(cleaner.plot())
    
    # war dates
    war_dates = enums.WAR_DATES.get(country)
    if war_dates:
        # set war dates before resampling 
        cleaner.set_war_var(war_dates)
        
    ## resample and visualize
    cleaner.resample('M')
    st.markdown(f'## resampled data:')
    resampled_plot = cleaner.plot(use_resampled=True)
    st.pyplot(resampled_plot)

    if war_dates:
        st.markdown(f'## periods of war in {country}:')
        # for start, end in war_dates:
        #     st.write(f'{start} to {end}')
            
        st.pyplot(cleaner.plot_war())
    else:
        st.write(f'no timeline of conflict history found for {country}')
        st.pyplot(cleaner.plot_war(use_resampled=True))
    

## ----------------------------------------------
## SEARCH
## ----------------------------------------------
col1, col2 = st.columns([3, 1]) 

with col1:
    search_keyword = st.text_input("search by country name")
with col2:
    # Button to generate the network
    search_button = st.button('search', key='search')

if search_button:
    country = yf.fuzzy_match(search_keyword)
    if country is not None:
        st.markdown(f"#### results of your input: {country}")
        cleaner = yf.UCDPCleaner(country)
        open_viz(cleaner)
    else:
        st.markdown(f"#### '{search_keyword}' is not currently supported. please try again!")

