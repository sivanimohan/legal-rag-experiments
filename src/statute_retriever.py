import json
import os
import requests

def query_hf_inference(prompt, model="mistralai/Mistral-7B-Instruct-v0.2"):
    HF_TOKEN = os.getenv("HF_TOKEN")  # Set this in your environment!
    API_URL = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 256,
            "return_full_text": False
        }
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code != 200:
        raise Exception(f"HF API error: {response.status_code} {response.text}")
    return response.json()[0]["generated_text"]

def load_statutes(path):
    with open(path, 'r') as f:
        return json.load(f)

def load_scenarios(path):
    with open(path, 'r') as f:
        return json.load(f)

def hf_retrieve_statutes(scenario_text, statutes):
    statutes_str = "\n".join([f"{i+1}. {s['title']}: {s['text']}" for i, s in enumerate(statutes)])
    prompt = f"""
Given the following statutes:

{statutes_str}

And the scenario: "{scenario_text}"

List the numbers of the statutes that are most relevant to the scenario. Only output the numbers, comma-separated if more than one.
"""
    result = query_hf_inference(prompt)
    relevant_numbers = []
    for part in result.replace(",", " ").split():
        if part.strip().isdigit():
            relevant_numbers.append(int(part))
    return [statutes[i-1] for i in relevant_numbers if 0 < i <= len(statutes)]

if __name__ == "__main__":
    statutes = load_statutes("data/statute_laws.json")
    scenarios = load_scenarios("data/test_scenarios.json")
    for scenario in scenarios:
        print(f"\nScenario: {scenario['facts']}")
        matched = hf_retrieve_statutes(scenario['facts'], statutes)
        print("Relevant Statutes:")
        for s in matched:
            print(f"  {s['id']}: {s['title']}")