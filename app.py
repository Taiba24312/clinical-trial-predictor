import streamlit as st
import joblib
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

# ===== PAGE CONFIG =====
st.set_page_config(page_title="Clinical Trial Predictor", layout="wide")

# ===== LOAD MODEL =====
model = joblib.load("clinical_trial_final_model_log_transform.pkl")

# ===== LOAD BERT =====
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
bert_model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
bert_model.eval()

# ===== MEAN POOLING =====
def mean_pooling(outputs, attention_mask):
    token_embeddings = outputs.last_hidden_state
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = torch.sum(token_embeddings * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts

# ===== PIPELINE =====
def prepare_input(summary, eligibility, condition, intervention, phase, enrollment):

    text = summary + " " + eligibility + " " + condition + " " + intervention

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    with torch.no_grad():
        outputs = bert_model(**inputs)

    embedding = mean_pooling(outputs, inputs["attention_mask"]).numpy()

    phase_map = {"PHASE1":1, "PHASE2":2, "PHASE3":3, "PHASE4":4}
    phase_num = phase_map[phase]

    enrollment = np.log1p(enrollment)

    X = np.zeros((1, 2289))
    X[:, :768] = embedding
    X[:, 768] = phase_num
    X[:, 769] = enrollment

    return X

# ===== UI =====
st.title("🧬 Clinical Trial AI Predictor")

summary = st.text_area("Summary")
eligibility = st.text_area("Eligibility")
condition = st.text_input("Condition")
intervention = st.text_input("Intervention")

phase = st.selectbox("Phase", ["PHASE1","PHASE2","PHASE3","PHASE4"])
enrollment = st.slider("Enrollment", 0, 1000, 100)

if st.button("Predict"):

    X = prepare_input(summary, eligibility, condition, intervention, phase, enrollment)
    prob = model.predict_proba(X)[0][1]

    st.metric("Success Probability", f"{prob:.2f}")
    st.progress(float(prob))

    if prob > 0.75:
        st.success("High success likelihood")
    elif prob > 0.5:
        st.warning("Moderate success likelihood")
    else:
        st.error("Low success likelihood")
