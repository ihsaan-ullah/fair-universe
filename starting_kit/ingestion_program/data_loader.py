import json 
import os 

def get_settings_from_json(dir='./'):

    json_dicts = []
    json_files = [f for f in os.listdir(dir) if f.endswith(".json")]
    for index, _ in enumerate(json_files):
        json_file = "data_"+str(index+1)+".json"
        json_dicts.append(json.load(open(os.path.join(dir, json_file))))

    return json_dicts
