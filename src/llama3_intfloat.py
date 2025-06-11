import json
import os
import requests
import re
from sentence_transformers import SentenceTransformer, util

# Dense Retriever (using e.g. 'intfloat/e5-large-v2' - not previously used above)
class E5DenseRetriever:
    def __init__(self, statutes, model_name="intfloat/e5-large-v2"):
        self.statutes = statutes
        self.corpus = [
            f"{s['title']} {s['text']}" for s in statutes
        ]
        self.model = SentenceTransformer(model_name)
        self.doc_embeds = self.model.encode(self.corpus, convert_to_tensor=True)

    def retrieve(self, scenario_text, top_k=5):
        query_embed = self.model.encode(scenario_text, convert_to_tensor=True)
        hits = util.semantic_search(query_embed, self.doc_embeds, top_k=top_k)[0]
        return [self.statutes[hit['corpus_id']] for hit in hits]

def query_groq_inference(prompt, model="llama3-70b-8192", max_new_tokens=1024):
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

def groq_generate_arguments(scenario, statutes, model="llama3-70b-8192", max_new_tokens=1024):
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
    retriever = E5DenseRetriever(statutes, model_name="intfloat/e5-large-v2")
    for scenario in scenarios:
        print(f"\nScenario: {scenario['facts']}")
        relevant_statutes = retriever.retrieve(scenario['facts'])
        print("Relevant Statutes (E5 Dense):")
        for s in relevant_statutes:
            print(f"  {s['id']}: {s['title']}")
        if not relevant_statutes:
            print("No relevant statutes found.\n")
        arguments = groq_generate_arguments(scenario, relevant_statutes)
        print("Generated Arguments:\n", arguments)