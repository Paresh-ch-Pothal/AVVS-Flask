import pickle
import json

with open("voter_data.json", "r") as file:
    VOTER_DATA = json.load(file)

print(len(VOTER_DATA))


