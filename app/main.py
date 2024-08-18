from transformers import pipeline

# You can check any other model in the Hugging Face Hub. In my case I chose this one to classify text by positive and negative sentiment. 
pipe = pipeline(model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

print(pipe("I Love You And You are So great!"))
# Output[{'label': 'POSITIVE', 'score': 0.9998719692230225}]