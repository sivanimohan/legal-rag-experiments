import json
import os
import requests
import re

# Simple keyword overlap retriever for 100% determinism
class SimpleKeywordRetriever:
    def __init__(self, statutes):
        self.statutes = statutes

    def retrieve(self, scenario_text, top_k=5):
        scenario_keywords = set(re.findall(r'\b\w+\b', scenario_text.lower()))
        scored = []
        for stat in self.statutes:
            text = (stat.get('title', '') + " " + stat.get('text', '')).lower()
            score = sum(1 for word in scenario_keywords if word in text)
            scored.append((score, stat))
        # Sort by score descending and select top_k
        top_statutes = [stat for score, stat in sorted(scored, key=lambda x: x[0], reverse=True)[:top_k] if score > 0]
        return top_statutes

def query_groq_inference(prompt, model="mistral-saba-24b", max_new_tokens=1024):
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

def groq_generate_arguments(scenario, statutes, model="mistral-saba-24b", max_new_tokens=1024):
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
    retriever = SimpleKeywordRetriever(statutes)
    for scenario in scenarios:
        print(f"\nScenario: {scenario['facts']}")
        # Simple Keyword Retriever
        relevant_statutes = retriever.retrieve(scenario['facts'])
        print("Relevant Statutes (Keyword Overlap):")
        for s in relevant_statutes:
            print(f"  {s['id']}: {s['title']}")
        if not relevant_statutes:
            print("No relevant statutes found.\n")
        # Use relevant statutes from keyword retriever for argument generation
        arguments = groq_generate_arguments(scenario, relevant_statutes)
        print("Generated Arguments:\n", arguments)