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

def build_argument_prompt(scenario, statutes):
    statutes_texts = "\n".join([f"{s['title']}: {s['text']}" for s in statutes])
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

def hf_generate_arguments(prompt):
    return query_hf_inference(prompt)

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
        relevant_statutes = hf_retrieve_statutes(scenario['facts'], statutes)
        if not relevant_statutes:
            print("No relevant statutes found.")
            continue
        prompt = build_argument_prompt(scenario, relevant_statutes)
        arguments = hf_generate_arguments(prompt)
        print("Generated Arguments:\n", arguments)