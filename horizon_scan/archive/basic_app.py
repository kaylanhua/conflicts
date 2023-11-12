import streamlit as st
import streamlit.components.v1 as components

# streamlit run app.py

st.title('kayla\'s network analysis')
st.write('an exploration of machine learning and conflict modeling')

# Load the HTML file
html_file = open('graph54.html', 'r', encoding='utf-8')
source_code = html_file.read() 
components.html(source_code, height=750)