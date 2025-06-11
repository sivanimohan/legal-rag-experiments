# legalRagExperiments
# Legal RAG Experiments

**Repository:** [sivanimohan/legalRagExperiments]  
**Language:** Python

This repository explores and benchmarks Retrieval-Augmented Generation (RAG) architectures for legal argument generation, focusing on the interplay between multiple classical and neural retrievers and a range of state-of-the-art LLMs. The aim is to empirically evaluate how retrieval strategies and LLM architectures impact the relevance and quality of generated legal arguments, given realistic legal scenarios and a statute corpus.

---

## 🏗️ Pipeline Overview

1. **Legal Scenario Input:**  
   Fact patterns are fed into the system as queries.

2. **Statute Retrieval:**  
   The query is processed using a selected retriever (lexical or neural/dense) to identify top-N relevant statutes from a JSON corpus.

3. **LLM Argumentation:**  
   The retrieved statutes and scenario are formatted into a prompt and sent to an LLM, which generates plaintiff and defendant arguments.

4. **Evaluation:**  
   Outputs are analyzed for legal accuracy, depth of reasoning, and adaptability to context.

---

## 🔍 Retrievers (ML/IR Perspective)

| Retriever        | Type      | Model/Algorithm         | Embedding Space | Pros                                | Cons                          |
|------------------|-----------|------------------------|-----------------|-------------------------------------|-------------------------------|
| **Keyword**      | Lexical   | Direct word overlap    | N/A             | Fast, no training required          | No semantic understanding     |
| **TF-IDF**       | Lexical   | tf-idf vectorizer      | Sparse          | Simple, interpretable               | Ignores context/synonyms      |
| **BM25**         | Lexical   | Okapi BM25             | Sparse          | Stronger weighting, open-source     | Still bag-of-words            |
| **Dense/MiniLM** | Neural    | all-MiniLM-L6-v2       | Dense           | Semantic similarity, small/fast     | May miss fine legal nuance     |
| **Dense/BGE**    | Neural    | BAAI/bge-base-en-v1.5  | Dense           | SOTA for retrieval, multi-lingual   | Large model, inference cost    |
| **Dense/E5**     | Neural    | intfloat/e5-large-v2   | Dense           | SOTA, robust on paraphrase/context  | Large, needs GPU, longer setup|

**Retriever Selection Rationale:**
- **Lexical retrievers** serve as baselines, representing traditional IR.  
- **Dense retrievers** use neural embedding models trained for semantic textual similarity, crucial for legal text where meaning often trumps surface form.
- **BGE/E5** are recent, strong models for English retrieval, tested here for legal domain transferability.

---

## 🧠 LLMs (ML Perspective)

| Model Name                              | Family        | Parameters | Tuning         | Context Window | Notable ML Features                  |
|----------------------------------------- |--------------|------------|----------------|---------------|--------------------------------------|
| **llama3-70b-8192**                      | Llama 3      | 70B        | Inst. & RLHF   | 8192 tokens    | Strong context retention, coherence  |
| **llama-3.3-70b-versatile**              | Llama 3      | 70B        | Versatile finetune | 8192 tokens| Multi-task, robust to prompt variety |
| **mistral-saba-24b**                     | Mistral      | 24B        | SFT/RLHF       | 32K tokens?    | Fast, structured output, lower cost  |
| **meta-llama/llama-4-maverick-17b-128e** | Llama 4      | 17B        | Instruct-tuned | 128K tokens    | Next-gen, safety, better reasoning   |
| **deepseek-r1-distill-llama-70b**        | Deepseek     | 70B        | Distilled      | 32K tokens?    | Efficient, distilled for inference   |

**LLM Selection Rationale:**
- **Parameter count** relates to reasoning depth and context integration.
- **Instruction tuning / RLHF** is critical for legal, multi-step prompting.
- **Latest-gen (Llama 4 Maverick, Deepseek-distill)** are tested for improved factuality, controllability, and safety.
- **Context window** is crucial for handling real-world legal input/output lengths.

---

## 🧪 Example Workflow

1. **Scenario:**  
   "Deceased P1 was married to P2 and allegedly ill-treated by her parents-in-law..."

2. **Retriever:**  
   E.g., E5 Dense retrieves statutes on 'Dowry death', 'Murder', 'Abetment', etc.

3. **Prompt Construction:**  
   Scenario and statutes are composed into a structured prompt for the LLM.

4. **LLM Output:**  
   The LLM generates arguments for both sides, referencing statutes, facts, and legal principles.

---

## 📈 Detailed Final Analysis (ML Focus)

### Retriever Analysis

- **Lexical retrievers** (Keyword, TF-IDF, BM25) are fast, interpretable, and good for explicit matches, but lack semantic generalization—leading to poor recall for paraphrased or contextually similar statutes.
- **Dense retrievers** (MiniLM, BGE, E5) use neural encoders to project texts into a shared embedding space. These capture semantics, enabling retrieval of statutes even when surface forms differ. BGE/E5 performed best in recall and relevance, especially for complex legal fact patterns.
- **Dense retrievers** require more resources (GPU for large models) and entail a setup cost but are critical for high-quality legal RAG.

### LLM Analysis

- **deepseek-r1-distill-llama-70b** and **llama3-70b-8192** consistently produced the most nuanced and contextually accurate legal arguments, justifying their higher parameter count and advanced tuning.
- **meta-llama/llama-4-maverick-17b-128e-instruct** balances new-generation reasoning and safety with efficient resource usage, yielding strong argument structure and adaptivity.
- **mistral-saba-24b** offers fast, concise outputs suitable for rapid prototyping or lower-complexity use cases but can lack deep legal nuance.
- **llama-3.3-70b-versatile** serves as a reliable, multi-task baseline, with solid output but sometimes less tailored than the larger/distilled models.

### ML Takeaway

- **Best Legal RAG Quality:**  
  Dense neural retrievers (E5/BGE) + large, instruction-tuned LLMs (Deepseek, Llama3-70B, Llama 4 Maverick).
- **Best for Speed/Scale:**  
  Lexical retrievers + mid-sized LLMs (Mistral, Llama-3.3 Versatile).
- **Optimal for Legal Domain:**  
  Dense retrieval is essential for semantic, context-rich legal scenarios; LLMs over 17B parameters with instruction tuning provide the best argument quality.

---

## 🚀 Getting Started

1. Install Python requirements:  
   `pip install sentence-transformers rank_bm25 requests`
2. Place statute corpus and scenarios in the `data/` folder (see example scripts).
3. Run your experiment scripts, choosing retriever and LLM as needed.

---

## 📚 References

- [Llama 3](https://github.com/meta-llama/llama3)
- [Sentence Transformers](https://www.sbert.net/)
- [BAAI BGE Models](https://huggingface.co/BAAI/bge-base-en-v1.5)
- [intfloat/e5 Models](https://huggingface.co/intfloat/e5-large-v2)
- [Mistral](https://mistralai.com/)
- [Deepseek](https://github.com/deepseek-ai/DeepSeek-LLM)

---

## 📝 License

MIT License
