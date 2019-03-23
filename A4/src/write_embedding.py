import json
from create_embedding import get_embedding

storepath = '../../../A4_data/embedding.txt'
with open(storepath, 'w') as file:
     file.write(json.dumps(get_embedding())) # use `json.loads` to do the reverse