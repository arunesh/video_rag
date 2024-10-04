import json


def transcribed_list_tostr(output_list):
    """ Converts the list of (timestamp) -> sentences to a readable string.
        Example:
           [{'timestamp': (0.0, 10.0), 'text': ' At Google, we are fully in our Gemini era.'}, ...
           ]
        Returns:
        string
    """
    output_str = ""
    for x in output_list:
        if "timestamp" in x:
            temp_str = f"{x['timestamp']} = {x['text']}"
            print(temp_str)
            output_str += (temp_str + "\n")
    return output_str

def transribed_list_to_jsonindex(transcribed_list, output_json_filepath):
    """ 
    Outputs the list as a json index for consumption later. A reverse index can be built in memory later.
    """
    result_list = transcribed_list
    # Write the data to a JSON file
    with open(output_json_filepath, 'w') as json_file:
        json.dump(result_list, json_file, indent=4)
    print(f"Index saved as f{output_json_filepath}")


def write_shardedlist_tofile(sharded_list, filepath):
    with open(filepath, "w") as f:
        for x in sharded_list:
            f.write(f"{x['text'] }\n")
    print(f"Wrote {filepath}")

def output_textonly_shards(transcribed_list, output_text_folder, shard_size=10):
    """
         Given the transcribed list of timestamps and sentences, this outputs it to a list of files
         with 'shard_size' sentences per file.
    """
    result_list = transcribed_list

    sharded_list = [result_list[i: i+shard_size] for i in range(0, len(result_list), shard_size)]
    shard_num = 0
    for x in sharded_list:
        # write to file
        write_shardedlist_tofile(x, f"{output_text_folder}/text_shard_{shard_num}.txt")
        shard_num += 1

