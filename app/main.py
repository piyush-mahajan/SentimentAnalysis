from transformers import pipeline

# You can check any other model in the Hugging Face Hub. In my case I chose this one to classify text by positive and negative sentiment. 
pipe = pipeline(model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
def generate_response(prompt:str):
   response = pipe("This is a great tutorial!")
   label = response[0]["label"]
   score = response[0]["score"]
   return f"The '{prompt}' input is {label} with a score of {score}"

print(generate_response("This tutorial is great!"))
# Output[{'label': 'POSITIVE', 'score': 0.9998719692230225}]