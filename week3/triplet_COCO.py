import json

# File path
PATH_ANNOTATIONS = '/ghome/group02/mcv/datasets/C5/COCO/mcv_image_retrieval_annotations.json'

# Load the dictionary from the file
with open(PATH_ANNOTATIONS, 'r') as file:
    annotations = json.load(file)

train = annotations['train']
database = annotations['database']
val = annotations['val']
test = annotations['test']

for image_id, labels in val.items():
    print(labels)


