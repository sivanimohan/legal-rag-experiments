import json
import os
import requests
import re
from rank_bm25 import BM25Okapi

# BM25 Retriever for 100% consistent and efficient retrieval
class BM25Retriever:
    def __init__(self, statutes):
        self.statutes = statutes
        self.corpus = [
            f"{s['title']} {s['text']}" for s in statutes
        ]
        self.tokenized_corpus = [self._tokenize(doc) for doc in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def _tokenize(self, text):
        # Simple whitespace and punctuation tokenizer
        return re.findall(r'\b\w+\b', text.lower())

    def retrieve(self, scenario_text, top_k=5):
        query = self._tokenize(scenario_text)
        scores = self.bm25.get_scores(query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [self.statutes[i] for i in top_indices if scores[i] > 0]

def query_groq_inference(prompt, model="llama-3.3-70b-versatile", max_new_tokens=1024):
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
    statutes_texts = "\n".join([f"{s['title']}: {s['text'][:300]}..." for s in statutes])  # Truncate statute text for brevity!
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

def groq_generate_arguments(scenario, statutes, model="llama-3.3-70b-versatile", max_new_tokens=1024):
    relevant_statutes = statutes
    prompt = build_argument_prompt(scenario, relevant_statutes)
    # Check token length
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
    retriever = BM25Retriever(statutes)
    for scenario in scenarios:
        print(f"\nScenario: {scenario['facts']}")
        # BM25 Retriever
        relevant_statutes_bm25 = retriever.retrieve(scenario['facts'])
        print("Relevant Statutes (BM25):")
        for s in relevant_statutes_bm25:
            print(f"  {s['id']}: {s['title']}")
        if not relevant_statutes_bm25:
            print("No relevant statutes found.\n")
        # Use relevant statutes from BM25 retriever for argument generation
        arguments = groq_generate_arguments(scenario, relevant_statutes_bm25)
        print("Generated Arguments:\n", arguments)