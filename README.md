# ***Video RAG***


Python project to explore building a RAG over a video.

## Week 3 update:

chainlit app that allows the VideoRag to download a video, create a MultiModalVectorDb and
answer questions.

chainlit run vr_chainlit_app.py -w

Conversation flow:
user: "i have a video url"
user: <provides a url>
user: answers yes for video processing
user: asks questions regarding the video.



### Datasets

- Google IO 2024 Keynote.

GPT4 for query and Claude as judge - accuracy: 26.67 percent.

GPT4 for query and judge - accuracy: 33.3 percent.

Claude for query and GPT4 as judge - accuracy: 80 percent

Claude for query and judge - accuracy: 76 percent.


### Conclusion

Claude is a better LLM for a question-answer engine compared to GPT4.

