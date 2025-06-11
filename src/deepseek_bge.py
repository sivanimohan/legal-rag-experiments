import json
import os
import requests
import re
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Best "dense" retriever (different model from any previously used): BAAI/bge-base-en-v1.5
class BGE_DenseRetriever:
    def __init__(self, statutes, model_name="BAAI/bge-base-en-v1.5"):
        self.model = SentenceTransformer(model_name)
        self.statutes = statutes
        self.statute_texts = [f"{s['title']}: {s['text']}" for s in statutes]
        self.embeddings = self.model.encode(self.statute_texts, convert_to_tensor=True)

    def retrieve(self, scenario_text, top_k=5, threshold=0.35):
        scenario_emb = self.model.encode(scenario_text, convert_to_tensor=True)
        cosine_scores = util.cos_sim(scenario_emb, self.embeddings)[0]
        top_k_actual = min(top_k, len(self.statutes))
        if top_k_actual == 0:
            return []
        top_indices = np.argsort(-cosine_scores.cpu().numpy())[:top_k_actual]
        relevant_statutes = []
        for idx in top_indices:
            if float(cosine_scores[idx]) >= threshold:
                relevant_statutes.append(self.statutes[idx])
        return relevant_statutes

def query_groq_inference(prompt, model="deepseek-r1-distill-llama-70b", max_new_tokens=1024):
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_API_KEY:
        raise Exception("Please set the GROQ_API_KEY environment variable.")
    API_URL = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful legal assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_new_tokens,
        "temperature": 0.7,
        "top_p": 0.95
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code != 200:
        raise Exception(f"Groq API error: {response.status_code} {response.text}")
    data = response.json()
    return data["choices"][0]["message"]["content"]

def estimate_tokens(text):
    # Roughly 4 characters per token
    return len(text) // 4

def build_argument_prompt(scenario, statutes):
    statutes_texts = "\n".join([f"{s['title']}: {s['text'][:300]}..." for s in statutes])
    prompt = f"""
Given the following legal scenario and relevant statutes, generate persuasive legal arguments for both the plaintiff and the defendant.

Scenario:
{scenario['facts']}

Relevant Statutes:
{statutes_texts}

Plaintiff Argument:
Defendant Argument:
"""
    return prompt

def groq_generate_arguments(scenario, statutes, model="deepseek-r1-distill-llama-70b", max_new_tokens=1024):
    relevant_statutes = statutes
    prompt = build_argument_prompt(scenario, relevant_statutes)
    if estimate_tokens(prompt) > 30000:
        print("Prompt too long even after filtering statutes! Truncating further.")
        relevant_statutes = relevant_statutes[:2]
        prompt = build_argument_prompt(scenario, relevant_statutes)
    return query_groq_inference(prompt, model=model, max_new_tokens=max_new_tokens)

def load_statutes(path):
    with open(path, 'r') as f:
        return json.load(f)

def load_scenarios(path):
    with open(path, 'r') as f:
        return json.load(f)

if __name__ == "__main__":
    statutes = load_statutes("data/statute_laws.json")
    scenarios = load_scenarios("data/test_scenarios.json")
    retriever = BGE_DenseRetriever(statutes)
    for scenario in scenarios:
        print(f"\nScenario: {scenario['facts']}")
        relevant_statutes = retriever.retrieve(scenario['facts'])
        print("Relevant Statutes (BGE Dense Retriever):")
        for s in relevant_statutes:
            print(f"  {s['id']}: {s['title']}")
        if not relevant_statutes:
            print("No relevant statutes found.\n")
        arguments = groq_generate_arguments(scenario, relevant_statutes)
        print("Generated Arguments:\n", arguments)