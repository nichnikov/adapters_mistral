import os
import json
from pprint import pprint

with open(os.path.join("data", "ecommerce-faq.json")) as json_file:
    data = json.load(json_file)

# print(data)
pprint(data["questions"][0], sort_dicts=False)

