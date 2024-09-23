# Generate QnA dataset and eval for video doc.
# In this version, we only support the transcription part of the video.


from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader


# Load env vars. from .env file
load_dotenv()

# Load documents from the "data_split/" folder
docs = SimpleDirectoryReader("data_split").load_data()

print("number of docs:", len(docs))

from openai import OpenAI
import json

oai_client = OpenAI()


# Function to generate questions and answers
def generate_qa(prompt, text, temperature=0.2):   
    response = oai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": text}],
        temperature=temperature,
    )

    print("Getting responses -------------------------")
    
    print(response.choices[0].message.content)

    # Strip extraneous symbols from the response content
    content = response.choices[0].message.content.strip()
    
    # Remove potential JSON code block markers
    content = content.strip()
    if content.startswith('```'):
        content = content.split('\n', 1)[-1]
    if content.endswith('```'):
        content = content.rsplit('\n', 1)[0]
    content = content.strip()
    
    # Attempt to parse the cleaned content as JSON
    try:
        parsed_content = json.loads(content.strip())
        return parsed_content
    except json.JSONDecodeError:
        print("Error: Unable to parse JSON. Raw content:")
        print(content)
        return []

factual_prompt = """
You are an expert educational content creator tasked with generating factual questions and answers based on the following transcript of a keynote session from Google I/O 2024. These questions should focus on retrieving specific details, figures, definitions, and key facts from the text.

Instructions:

- Generate **5** factual questions, each with a corresponding **expected_output**.
- Ensure all questions are directly related to the document excerpt.
- Present the output in the following structured JSON format:

[
  {
    "question": "What is the main purpose of the project described in the document?",
    "expected_output": "To develop a new framework for data security using AI-powered tools."
  },
  {
    "question": "Who authored the report mentioned in the document?",
    "expected_output": "Dr. Jane Smith."
  }
]
"""


# Generate dataset
import os
import json

dataset_file = 'video_rag_googleio_qa_dataset_2.json'

if os.path.exists(dataset_file):
    # Load dataset from local file if it exists
    with open(dataset_file, 'r') as f:
        dataset = json.load(f)
else:
    # Generate dataset if local file doesn't exist
    dataset = []
    for doc in docs:
        print("Doc -------", doc.get_doc_id())
        print("Doc length = ", len(doc.text))
        print("Doc metadata:", doc.metadata)
        qa_pairs = generate_qa(factual_prompt, doc.text, temperature=0.2)
        dataset.extend(qa_pairs)
    
    # Write dataset to local file
    with open(dataset_file, 'w') as f:
        json.dump(dataset, f)

        
# Note: we're choosing to create the dataset in Langfuse below, but it's equally easy to create it in another platform.
input("Press Enter to upload to Langfuse ...")
from langfuse import Langfuse
langfuse = Langfuse()

dataset_name = "video_rag_googleio_qa_pairs_6"
langfuse.create_dataset(name=dataset_name);

for item in dataset:
  langfuse.create_dataset_item(
      dataset_name=dataset_name,
      input=item["question"],
      expected_output=item["expected_output"]
)
