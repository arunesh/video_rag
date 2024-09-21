import streamlit as st

st.title('Video RAG Demo')


with st.form("Video Input:"):
  st.write("Provide a YouTube URL or upload a video:")
  my_number = st.slider('Pick a number', 1, 10)
  my_color = st.selectbox('Pick a color', ['red','orange','green','blue','violet'])
  st.form_submit_button('Submit')

