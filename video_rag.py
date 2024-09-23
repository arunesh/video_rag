import streamlit as st

from ingest_video import *

st.title('Video RAG Playground')

output_folder = "./temp/video_data/"

if 'video_url' not in st.session_state:
  st.session_state['video_url'] = ""

with st.form("Video Input:"):
  st.write("Provide a YouTube URL or upload a video:")
  my_number = st.slider('Pick a number', 1, 10)
  st.session_state.video_url = st.text_input("Paste a YouTube URL")
  my_color = st.selectbox('Pick a color', ['red','orange','green','blue','violet'])
  submit = st.form_submit_button('Submit')

if st.session_state.video_url != "":
  st.video(st.session_state.video_url)

if submit:
  st.write("Triggering video processing..")
  video = Video.from_url(st.session_state.video_url)
  video.download(output_folder)
  (v, a, t) = video.process_video()
  st.write(f"Video saved as {v}, audio as {a}, text as {t}")
  print(f"Video saved as {v}, audio as {a}, text as {t}")
  
