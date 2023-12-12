import streamlit as st

input = st.text_input(label="1",label_visibility="hidden",value="")
st.button("search")
col1,col2 = st.columns([0.5,0.5])
