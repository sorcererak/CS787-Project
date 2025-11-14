#import statements
import os, re, json, time, requests
import pandas as pd
from tqdm.auto import tqdm

import google.generativeai as genai
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login

user_secrets = UserSecretsClient()
login(token=user_secrets.get_secret("HF_TOKEN"))

#api keys and data paths
SERPER_API_KEY = user_secrets.get_secret("SERPER_API_KEY")   # Serper key (optional)
GEMINI_API_KEY = user_secrets.get_secret("GEMINI_API_KEY")   # Gemini key (required)
DATA_PATH = "/kaggle/input/medmcqa-100/medmcqa_100.csv"

#Column names in your MedMCQA CSV
QUESTION_COL = "question"
OPTION_COLS = ["opa", "opb", "opc", "opd"]  
ANSWER_COL = "cop"                           

# slm
LOCAL_MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
DEVICE_MAP = "auto"

#gemini llm
GEMINI_MODEL_NAME = "gemini-2.5-flash-lite"

# web search
MAX_WEB_RESULTS = 5          
WEB_TIMEOUT = 10

# eval
MAX_QUESTIONS_EVAL = 100      
PRINT_FIRST_N_DEBUG = 3      
SLEEP_BETWEEN_GEMINI_CALLS = 0.3


# init models

# Gemini
gemini_model = None
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
    print("Gemini model initialised.")
else:
    print("WARNING: GEMINI_API_KEY not set, Gemini LLM disabled.")

# SLM (local 3B)
print("Loading local SLM (this can take a bit)...")
bnb_config = BitsAndBytesConfig(load_in_4bit=True)
slm_tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_NAME)
slm_model = AutoModelForCausalLM.from_pretrained(
    LOCAL_MODEL_NAME,
    device_map=DEVICE_MAP,
    quantization_config=bnb_config
)
print("Local SLM loaded.")


# helper: SLM generate

def slm_generate(prompt, max_new_tokens=64):
    """Generate text from local SLM."""
    inputs = slm_tokenizer(prompt, return_tensors="pt").to(slm_model.device)
    out_ids = slm_model.generate(**inputs, max_new_tokens=max_new_tokens)
    text = slm_tokenizer.decode(out_ids[0], skip_special_tokens=True)
    return text.strip()


def web_search(query, num_results=2):
    """
    Web search via Serper (if key) else fallback dummy snippet.
    Returns list of {'snippet', 'url'}.
    """
    num_results = min(num_results, MAX_WEB_RESULTS)
    results = []

    if SERPER_API_KEY:
        url = "https://google.serper.dev/search"
        headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
        payload = {"q": query}
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=WEB_TIMEOUT)
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


# slm decompose

def slm_decompose_queries(question):
    """
    Use SLM to decompose the original question into exactly three sub-queries:
    WHY, WHAT, HOW.
    Returns a dict: {"WHY": ..., "WHAT": ..., "HOW": ...}
    """
    prompt = (
        "You are a query decomposition assistant for medical entrance / board-style\n"
        "multiple-choice questions.\n"
        "Given a clinical or biomedical MCQ, rewrite it into exactly three concise\n"
        "sub-questions:\n"
        "1) a 'WHY' question (reason / pathophysiology / rationale)\n"
        "2) a 'WHAT' question (diagnosis / definition / key concept)\n"
        "3) a 'HOW' question (mechanism, treatment approach, or investigation)\n\n"
        "Return them in the following format exactly:\n"
        "WHY: <why_question>\n"
        "WHAT: <what_question>\n"
        "HOW: <how_question>\n\n"
        f"Original question: {question}\n\n"
        "Decomposed queries:"
    )

    text = slm_generate(prompt, max_new_tokens=128)

    why = what = how = question  # fallbacks
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


# Slm hints built

def slm_build_hints(question, tagged_snippets):
    """
    SLM = retriever + summariser.
    Given the question and tagged web snippets (with WHY/WHAT/HOW tags),
    SLM builds a small set of factual hints.
    """
    prompt = (
        "You are helping with a medical multiple-choice exam (similar to MedMCQA).\n"
        "You are given one clinical / biomedical question and some web snippets that\n"
        "were retrieved using different sub-queries (WHY / WHAT / HOW).\n"
        "Your job is to extract 4–8 short factual bullet points that are helpful\n"
        "for reasoning about the correct answer.\n"
        "IMPORTANT: Do NOT answer the question yourself. Only write the factual points.\n"
        "Each bullet point should be a single, clear statement about medicine, biology,\n"
        "pharmacology, diagnostics, or treatment.\n\n"
        f"Question: {question}\n\n"
        "Web snippets (with tags):\n"
    )
    for i, s in enumerate(tagged_snippets, 1):
        prompt += f"[{i}] {s['snippet']}\n"

    prompt += (
        "\nNow write 4–8 bullet points, each on a new line, starting with '- '.\n"
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


#MEDMCQA prompting Gemini

def gemini_mcq_answer_with_hints(question, options_dict, facts):
    """
    LLM = final answerer for MedMCQA.
    Gemini sees the medical question + options + OPTIONAL SLM-provided hints
    and chooses A/B/C/D. It may ignore the hints if they seem unhelpful.
    """
    if gemini_model is None:
        return "A"  # fallback if Gemini not configured

    text = (
        "You are solving a medical multiple-choice question from a challenging exam\n"
        "(similar to MedMCQA / medical entrance tests).\n"
        "Your goal is to choose the single best answer among A, B, C, and D.\n\n"
        "You are given:\n"
        "  • the question (which may describe a clinical scenario),\n"
        "  • four answer options,\n"
        "  • OPTIONAL factual hints produced by a smaller helper model from web search.\n\n"
        "Use the following strategy:\n"
        "  1. Carefully read the question, paying attention to age, symptoms, labs,\n"
        "     and key clinical findings.\n"
        "  2. Use your own medical knowledge and reasoning to understand what is being asked\n"
        "     (diagnosis, mechanism, investigation, treatment, etc.).\n"
        "  3. Treat the hints as optional support: they may be useful but can be\n"
        "     incomplete, noisy, or partially irrelevant.\n"
        "     • If the hints are consistent with the question and your reasoning,\n"
        "       you may use them to refine or double-check your choice.\n"
        "     • If the hints seem inconsistent or unhelpful, ignore them and answer\n"
        "       purely from your own understanding.\n"
        "  4. Quietly eliminate implausible options using medical principles and\n"
        "     standard-of-care knowledge.\n"
        "  5. Select the single best remaining option.\n\n"
        "Important formatting instruction:\n"
        "  • Do NOT show your reasoning or explanation.\n"
        "  • Return ONLY a single capital letter: A, B, C, or D.\n\n"
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
            "No explicit hints are provided; rely entirely on the question, options,\n"
            "and your own medical knowledge.\n"
        )

    text += "\nAnswer (ONLY one letter A, B, C, or D):"

    try:
        resp = gemini_model.generate_content(text)
        out_text = resp.text.strip()
    except Exception as e:
        print("Gemini error while answering MedMCQA MCQ:", e)
        out_text = ""

    time.sleep(SLEEP_BETWEEN_GEMINI_CALLS)

    m = re.search(r"[ABCD]", out_text)
    if m:
        return m.group(0)
    return "A"


# mapping helper

def normalize_gold_label(raw):
    """
    MedMCQA 'cop' is often 0/1/2/3 (index of correct option).
    This converts various encodings (0-3, 1-4, A-D) into 'A'..'D'.
    """
    s = str(raw).strip()


    if s.upper() in ["A", "B", "C", "D"]:
        return s.upper()


    if s in ["0", "1", "2", "3"]:
        mapping = {"0": "A", "1": "B", "2": "C", "3": "D"}
        return mapping[s]


    if s in ["1", "2", "3", "4"]:
        mapping = {"1": "A", "2": "B", "3": "C", "4": "D"}
        return mapping[s]

 
    return s.upper()


# per-question pipeline

def predict_mcq_for_row(row, verbose=False):
    """
    Pipeline for a single MedMCQA question:

      1) Use SLM to decompose question into WHY / WHAT / HOW sub-queries.
      2) For each sub-query, retrieve 2 web snippets (total ~6 snippets).
      3) Tag snippets with the sub-query type and aggregate them.
      4) SLM builds structured hints from all tagged snippets.
      5) Gemini (LLM) answers using question + options + OPTIONAL SLM hints.
      6) Final prediction = Gemini answer (no override).
    """
    question = str(row[QUESTION_COL])
    options = {
        "A": str(row[OPTION_COLS[0]]),
        "B": str(row[OPTION_COLS[1]]),
        "C": str(row[OPTION_COLS[2]]),
        "D": str(row[OPTION_COLS[3]]),
    }

    # 1) Decompose question into WHY/WHAT/HOW
    sub_queries = slm_decompose_queries(question)

    # 2) Retrieve web snippets for each sub-query (2 per query)
    tagged_snippets = []
    for tag in ["WHY", "WHAT", "HOW"]:
        subq = sub_queries[tag]
        sub_snips = web_search(subq, num_results=2)
        for s in sub_snips:
            tagged_snippets.append({
                "snippet": f"[{tag}] {s['snippet']}",
                "url": s["url"]
            })

    # 3) SLM builds hints from aggregated tagged snippets
    facts = slm_build_hints(question, tagged_snippets)

    # 4) Gemini answers using OPTIONAL SLM hints
    ans_llm = gemini_mcq_answer_with_hints(question, options, facts)
    final_pred = ans_llm

    if verbose:
        print("\n" + "=" * 80)
        print(f"Question: {question}\n")
        print("Options:")
        for lab in ["A", "B", "C", "D"]:
            print(f"{lab}. {options[lab]}")
        print("\nDecomposed sub-queries:")
        for tag in ["WHY", "WHAT", "HOW"]:
            print(f"  {tag}: {sub_queries[tag]}")
        print("\nTagged web snippets:")
        for i, s in enumerate(tagged_snippets, 1):
            print(f"  [{i}] {s['snippet'][:160]}")
        print("\nSLM factual hints:")
        for f in facts:
            print(f"  - {f}")
        print("\nLLM (Gemini) final answer:", final_pred)

    return final_pred, ans_llm, sub_queries, facts


# final run

df = pd.read_csv(DATA_PATH)

print("Columns in MedMCQA csv:", list(df.columns))

total = min(MAX_QUESTIONS_EVAL, len(df))
predictions = []
gold_labels = []
correct_flags = []

llm_answers = []
all_sub_queries = []
all_facts = []

for idx in tqdm(range(total), desc="Evaluating MedMCQA-100 (SLM-decomp + web + hints + LLM)"):
    row = df.iloc[idx]
    gold_raw = row[ANSWER_COL]
    gold = normalize_gold_label(gold_raw)
    gold_labels.append(gold)

    verbose = idx < PRINT_FIRST_N_DEBUG

    pred, ans_llm, sub_queries, facts = predict_mcq_for_row(row, verbose=verbose)

    predictions.append(pred)
    llm_answers.append(ans_llm)
    all_sub_queries.append(sub_queries)
    all_facts.append(facts)

    is_corr = int(pred == gold)
    correct_flags.append(is_corr)

    if not verbose:
        print(f"Q{idx+1}: gold={gold} (raw={gold_raw}), pred={pred}, LLM={ans_llm}")

# save results

correct = sum(correct_flags)
accuracy = correct / total
print(f"\nFinished {total} MedMCQA questions.")
print(f"Pipeline accuracy (LLM final, SLM = decomposition+retrieval+hinting): {accuracy:.3f}")

eval_df = df.iloc[:total].copy()
eval_df["gold_normalized"] = gold_labels
eval_df["pred"] = predictions
eval_df["correct"] = correct_flags
eval_df["llm_answer"] = llm_answers
eval_df["sub_queries"] = [json.dumps(q) for q in all_sub_queries]
eval_df["facts"] = [json.dumps(f) for f in all_facts]

save_path = "medmcqa_100_slm_decomp_web_hints_llm_final_results.csv"
eval_df.to_csv(save_path, index=False)
print(f"Saved MedMCQA results to: {save_path}")
