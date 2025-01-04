# model.py
from transformers import pipeline

# Load a pre-trained model from Hugging Face
model = pipeline("text-classification", model="distilbert-base-uncased")

def predict(text):
    return model(text)

if __name__ == "__main__":
    sample_text = "Hugging Face is amazing!"
    print(predict(sample_text))
