# CS787 Project — LLM Evaluation (Instruction vs Chain-of-Thought)

This repository contains code to evaluate Large Language Models (LLMs) on multiple
question-answering / reasoning benchmarks using two prompting styles:

1. **Instruction-based prompting**  
2. **Chain-of-Thought (CoT) prompting**

The goal is to compare how much *explicit reasoning in the prompt* helps across
different datasets.

---

## 🔍 Prompting Styles

### 1. Instruction-Based Prompting (`*_instruction*`)
- Direct, concise task description  
- Model is told **what to do**, not **how to reason**  
- No requirement to show intermediate steps  
- Fast, cheap, and serves as a **baseline**

### 2. Chain-of-Thought Prompting (`*_cot*`)
- Prompt explicitly asks the model to **think step-by-step**  
- Encourages reasoning, decomposition and explanation  
- Typically improves performance on:
  - Multi-step science questions  
  - Commonsense reasoning  
  - Hard multiple-choice questions  

---

## 📂 Repository Structure

Each dataset has **two versions** of the code: one for *instruction* prompting and
one for *CoT* prompting.

Examples (names may be `.py` or `.ipynb`):

- **ARC**
  - `arc_code_instruct.ipynb`
  - `arc_cot.py`
- **HellaSwag**
  - `hellaswag_code_instruction.ipynb`
  - `hellaswag_cot.py`
- **TruthfulQA**
  - `truthfulqa_code_instruction.py`
  - `truthfulqa_code_cot.ipynb`
- **GPQA**
  - `gpqa_code_instruction.ipynb`
  - `gpqa_cot.py`
- **MedMCQA**
  - `medmcqa_instruction.py`
  - `medmcqa_cot.py`
- **MMLU**
  - `mmlu_cot.py`
- **MUSR**
  - `musr_code_instruction.ipynb`
  - `musr_cot.py`
- **AVERIE / other datasets**
  - `averice_code_cot&instruct.ipynb` (combined notebook)

> In general, files containing **`instruction`** run the *instruction-style*
> prompts, and files containing **`cot`** run the *Chain-of-Thought* prompts.

---

## ▶️ How to Run

All scripts follow the same high-level flow:

1. Load dataset  
2. Build prompts (instruction or CoT)  
3. Query the LLM  
4. Parse predictions  
5. Compute accuracy  
6. Save a results file (CSV or similar)

### Example: Run CoT on ARC (Python script)

```bash
