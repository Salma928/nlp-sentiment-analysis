import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Load model
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("./model")
    model = AutoModelForSequenceClassification.from_pretrained("./model")
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# UI
st.title("Sentiment Analysis")
st.write("Multilingual sentiment analysis powered by XLM-RoBERTa")

text = st.text_area("Enter a sentence:", placeholder="This movie was absolutely fantastic!")

if st.button("Analyze"):
    if text.strip() == "":
        st.warning("Please enter a sentence.")
    else:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        pred = torch.argmax(probs).item()
        confidence = probs[0][pred].item()

        label = "Positive" if pred == 1 else "Negative"
        emoji = "😊" if pred == 1 else "😞"

        st.markdown(f"### {emoji} {label}")
        st.progress(confidence)
        st.write(f"Confidence: {confidence:.1%}")