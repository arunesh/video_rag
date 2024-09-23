import re
import math
import sys

def split_text_into_sentences(text):
    # Regular expression to match sentences ending with ., ?, or !
    sentence_endings = re.compile(r'(?<=[.!?]) +')
    sentences = sentence_endings.split(text)
    return sentences

def write_shards(sentences, num_shards, output_prefix="shard"):
    total_sentences = len(sentences)
    sentences_per_shard = math.ceil(total_sentences / num_shards)
    
    # Split sentences into roughly equal shards
    for i in range(num_shards):
        start_index = i * sentences_per_shard
        end_index = start_index + sentences_per_shard
        shard_sentences = sentences[start_index:end_index]
        
        # Write each shard into a new file
        with open(f"{output_prefix}_{i+1}.txt", "w") as shard_file:
            shard_file.write(" ".join(shard_sentences))

def split_text_file_into_shards(input_file, num_shards, output_prefix="shard"):
    # Read the entire text file
    with open(input_file, "r") as file:
        text = file.read()
    
    # Split the text into sentences
    sentences = split_text_into_sentences(text)
    
    # Split the sentences into shards and write them to new files
    write_shards(sentences, num_shards, output_prefix)


def run_main():
    split_text_file_into_shards(sys.argv[1], 5, sys.argv[2])

if __name__ == "__main__":
    run_main()