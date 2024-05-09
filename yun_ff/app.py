import streamlit as st
import streamlit.components.v1 as components
import yun_functions as yf
import enums 
import mpld3

# streamlit run app.py
st.set_page_config(layout="wide", page_title="Conflict Analysis", initial_sidebar_state="expanded", page_icon="🌍")
# st.set_page_config(layout="wide")


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
    # fig = cleaner.plot()
    # fig_html = mpld3.fig_to_html(fig)
    # components.html(fig_html, height=600)
    
    
    # war dates
    war_dates = enums.WAR_DATES.get(country)
    if war_dates:
        # set war dates before resampling 
        cleaner.set_war_var(war_dates)
        
    ## resample and visualize
    cleaner.resample('M') # TODO make this a user input
    st.markdown(f'### Graph of resampled data')
    resampled_plot = cleaner.plot(use_resampled=True)
    st.pyplot(resampled_plot)

    if war_dates:
        st.markdown(f'### Periods of war in {country}')
        for start, end in war_dates:
            st.write(f'From {start.strftime("%B %d, %Y")} to {end.strftime("%B %d, %Y")}')

   
    else:
        st.write(f'no timeline of conflict history found for {country}')
    
    st.pyplot(cleaner.plot_war())
    
def open_basics(country):
    raw_data = yf.UCDPCleaner(country).data
    

## ----------------------------------------------
## USER INPUTS 
## ----------------------------------------------

st.title('Civil Conflict Analysis')
st.write('An interactive exploration of conflict data & modeling')
st.markdown("---")


col1, _, col2, _, col3 = st.columns([4, 1, 10, 1, 4]) 

with col1:
    st.markdown('## Configurations')
    search_keyword = st.text_input("Search by country name ", "Sri Lanka")
    agg_len = st.text_input("Enter the aggregation length for the data: ", "M")
    window_len = st.text_input("Enter the window length, in months, for the training: ", "5")
    background_info = st.text_area("What additional background information on this conflict would you like? ", "What caused the first civil war?")

    st.markdown("---")
    model_selection = st.radio(
        "Select the model to use for forecasting:",
        ('Linear Regression', 'Random Forest', 'XGBoost', 'Elastic Net', 'LLM (in beta)')
    )
    search_button = st.button('Search', key='search')
    


with col2:
    
    if search_button:
        country = yf.fuzzy_match(search_keyword)
        if country is not None:
            st.markdown(f"## Showing results for {country}")
            cleaner = yf.UCDPCleaner(country)
            open_viz(cleaner)
        else:
            st.markdown(f"#### '{search_keyword}' is not currently supported. please try again!")

with col3:
    if not search_button:
        st.markdown('## About this project')
        st.write('This app is a work in progress. It is designed to help users explore conflict data and visualize trends over time. The data is sourced from the UCDP and is updated regularly.')
        st.write('For more information, please contact the developer at kaylahuang@college.harvard.edu')

    if search_button and country is not None:
        st.markdown(f'## Conflict background')
        background = yf.llm_country_info(country, more_info=background_info)
        st.write(background)
        open_basics(country)
    