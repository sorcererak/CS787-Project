import os
import re
import json
import time
import requests
import pandas as pd
from tqdm.auto import tqdm
import google.generativeai as genai
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login

user_secrets = UserSecretsClient()
login(token=user_secrets.get_secret("HF_TOKEN"))

SERPER_API_KEY = user_secrets.get_secret("SERPER_API_KEY")
GEMINI_API_KEY = user_secrets.get_secret("GEMINI_API_KEY")

path = "..."
model_name = "..."
DEVICE_MAP = "auto"
llm_name = "..."

web_results = 5
web_timeout = 10

total_eval_datapoints = 100
SLEEP_BETWEEN_GEMINI_CALLS = 0.3

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel(llm_name)
print("Gemini model initialised.")

print("Loading local SLM (this can take a bit)...")
bnb_config = BitsAndBytesConfig(load_in_4bit=True)
slm_tokenizer = AutoTokenizer.from_pretrained(model_name)
slm_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map=DEVICE_MAP,
    quantization_config=bnb_config,
)
print("Local SLM loaded.")

def slm_generate(prompt: str, max_new_tokens: int = 64) -> str:
    """Generate text from the local SLM."""
    inputs = slm_tokenizer(prompt, return_tensors="pt").to(slm_model.device)
    out_ids = slm_model.generate(**inputs, max_new_tokens=max_new_tokens)
    text = slm_tokenizer.decode(out_ids[0], skip_special_tokens=True)
    return text.strip()


def web_search(query: str, num_results: int = 2):
    num_results = min(num_results, web_results)
    results = []

    if SERPER_API_KEY:
        url = "https://google.serper.dev/search"
        headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
        payload = {"q": query}
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=web_timeout)
            data = r.json()
            if data.get("organic"):
                for item in data["organic"][:num_results]:
                    snippet = (item.get("snippet") or "").replace("\n", " ")
                    link = item.get("link") or ""
                    if snippet:
                        results.append({"snippet": snippet, "url": link})
        except Exception as e:
            print("Serper error:", e)

    if not results:
        results.append({"snippet": f"General information about: {query}", "url": ""})

    return results[:num_results]

def slm_decompose_queries(question):
    prompt = (
        "You are a query decomposition assistant for science exam questions.\n"
        "Given a multiple-choice question, rewrite it into exactly three concise\n"
        "sub-questions:\n"
        "1) a 'WHY' question\n"
        "2) a 'WHAT' question\n"
        "3) a 'HOW' question\n\n"
        "Return them in the following format exactly:\n"
        "WHY: <why_question>\n"
        "WHAT: <what_question>\n"
        "HOW: <how_question>\n\n"
        f"Original question: {question}\n\n"
        "Decomposed queries:"
    )
    text = slm_generate(prompt, max_new_tokens=128)
    why = what = how = question
    m_why = re.search(r"WHY:\s*(.+)", text, flags=re.IGNORECASE)
    m_what = re.search(r"WHAT:\s*(.+)", text, flags=re.IGNORECASE)
    m_how = re.search(r"HOW:\s*(.+)", text, flags=re.IGNORECASE)
    if m_why:
        why = m_why.group(1).strip()
    if m_what:
        what = m_what.group(1).strip()
    if m_how:
        how = m_how.group(1).strip()
    return {"WHY": why, "WHAT": what, "HOW": how}

def slm_build_hints(question: str, tagged_snippets):
    prompt = (
        "You are given a science question and some web snippets that were retrieved\n"
        "using different sub-queries (WHY / WHAT / HOW).\n"
        "Your job is to extract 4-8 short factual bullet points that are helpful "
        "for answering the question.\n"
        "IMPORTANT: Do NOT answer the question yourself. Only write the factual points.\n"
        "Each bullet point should be a single simple statement.\n\n"
        f"Question: {question}\n\n"
        "Web snippets (with tags):\n"
    )

    for i, s in enumerate(tagged_snippets, 1):
        prompt += f"[{i}] {s['snippet']}\n"

    prompt += (
        "\nNow write 4-8 bullet points, each on a new line, starting with '- '.\n"
        "Do not include anything else.\n"
    )

    text = slm_generate(prompt, max_new_tokens=192)

    facts = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("-"):
            fact = line.lstrip("-").strip()
            if fact:
                facts.append(fact)
    if not facts:
        facts = [s["snippet"] for s in tagged_snippets]

    return facts

def gemini_mcq_answer_with_hints(question, options_dict, facts):
    text = (
        "You are a highly capable science exam solver.\n"
        "Your goal is to choose the single best answer to the multiple-choice question.\n\n"
        "You are given:\n"
        "  i) the question\n"
        "  ii) four answer options (A, B, C, D)\n"
        "  iii) OPTIONAL factual hints produced by a smaller helper model from web search\n\n"
        "Use the following reasoning strategy:\n"
        "  1. Carefully read the question and options.\n"
        "  2. Think step-by-step and reason explicitly about what the question requires.\n"
        "  3. Integrate your own scientific knowledge.\n"
        "  4. Consider the hints as optional context — use them only if they align with your reasoning.\n"
        "  5. Eliminate wrong options one-by-one using logical deduction.\n"
        "  6. Arrive at the single best option.\n\n"
        "Important formatting instructions:\n"
        "  i) Show your full reasoning process step-by-step.\n"
        "  ii) After the reasoning, output the final answer in the format:\n"
        "        Final Answer: X\n"
        "      where X is one of A, B, C, or D.\n\n"
        f"Question: {question}\n\n"
        "Options:\n"
        f"A. {options_dict['A']}\n"
        f"B. {options_dict['B']}\n"
        f"C. {options_dict['C']}\n"
        f"D. {options_dict['D']}\n\n"
    )

    if facts:
        text += "Optional factual hints (from a smaller helper model):\n"
        for f in facts:
            text += f"- {f}\n"
    else:
        text += (
            "No explicit hints are provided; rely entirely on the question, options, "
            "and your own knowledge.\n"
        )

    text += "\nAnswer (ONLY one letter A, B, C, or D):"

    try:
        resp = gemini_model.generate_content(text)
        out_text = resp.text.strip()
    except Exception as e:
        print("Gemini error while answering MCQ:", e)
        out_text = ""

    time.sleep(SLEEP_BETWEEN_GEMINI_CALLS)
    m = re.search(r"[ABCD]", out_text)
    if m:
        return m.group(0)
    return "A"

def predict_mcq_for_row(row, q_col, opt_cols):
    question = str(row[q_col])
    options = {
        "A": str(row[opt_cols[0]]),
        "B": str(row[opt_cols[1]]),
        "C": str(row[opt_cols[2]]),
        "D": str(row[opt_cols[3]]),
    }

    sub_queries = slm_decompose_queries(question)
    tagged_snippets = []
    
    for tag in ["WHY", "WHAT", "HOW"]:
        subq = sub_queries[tag]
        sub_snips = web_search(subq, num_results=2)
        for s in sub_snips:
            tagged_snippets.append({
                "snippet": f"[{tag}] {s['snippet']}",
                "url": s["url"],
            })

    facts = slm_build_hints(question, tagged_snippets)
    ans_llm = gemini_mcq_answer_with_hints(question, options, facts)
    final_pred = ans_llm
    return final_pred, ans_llm, sub_queries, facts

df = pd.read_csv(path)
cols = list(df.columns)
q_col = cols[0]
opt_cols = cols[1:5]
ans_col = cols[5]

print("Columns:", cols)
print("Question column:", q_col)
print("Option columns:", opt_cols)
print("Answer column:", ans_col)

total = min(total_eval_datapoints, len(df))
predictions = []
gold_labels = []
correct_flags = []
llm_answers = []
all_sub_queries = []
all_facts = []

for idx in tqdm(range(total), desc="Evaluating ARC-100 (SLM-decomp + web + hints + LLM)"):
    row = df.iloc[idx]
    gold = str(row[ans_col]).strip().upper()
    gold_labels.append(gold)
    pred, ans_llm, sub_queries, facts = predict_mcq_for_row(row, q_col, opt_cols)
    predictions.append(pred)
    llm_answers.append(ans_llm)
    all_sub_queries.append(sub_queries)
    all_facts.append(facts)
    is_corr = int(pred == gold)
    correct_flags.append(is_corr)
    print(f"Q{idx + 1}: gold={gold}, pred={pred}, LLM={ans_llm}")

correct = sum(correct_flags)
accuracy = correct/total
print(f"\nFinished {total} questions.")
print(f"Pipeline accuracy (LLM final, SLM = decomposition+retrieval+hinting): {accuracy:.3f}")

eval_df = df.iloc[:total].copy()
eval_df["gold"] = gold_labels
eval_df["pred"] = predictions
eval_df["correct"] = correct_flags
eval_df["llm_answer"] = llm_answers
eval_df["sub_queries"] = [json.dumps(q) for q in all_sub_queries]
eval_df["facts"] = [json.dumps(f) for f in all_facts]

save_path = "arc_instruction.csv"
eval_df.to_csv(save_path, index=False)
print(f"Saved detailed results to: {save_path}")