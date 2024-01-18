import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import streamlit as st

st.title("Sentence Validity Checker")
sentences = st.text_input("Enter the Sentence")
tokenizer = AutoTokenizer.from_pretrained("Ashishkr/query_wellformedness_score")
model = AutoModelForSequenceClassification.from_pretrained("Ashishkr/query_wellformedness_score")

features = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
model.eval()
with torch.no_grad():
    scores = model(**features).logits
    score_value = round(scores.data[0].item(), 4)

st.write("Score ", score_value)
