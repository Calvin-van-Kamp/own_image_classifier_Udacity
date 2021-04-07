import json

def cat_to_name_dict(file = 'cat_to_name.json'):
  """
  Function that creates a dictionary from which the class names can be identified.
  Inputs:
      file: str, location of the json-stored classes, cat_to_name.json as default
  Returns:
      Dictionary of class numbers with respective class names
  """
  with open('cat_to_name.json', 'r') as f:
      name_dict = json.load(f)
  return name_dict