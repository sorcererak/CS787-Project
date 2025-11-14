# import statements
import os, re, json, time, requests
import pandas as pd
from tqdm.auto import tqdm
import google.generativeai as genai
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login

user_secrets = UserSecretsClient()
login(token=user_secrets.get_secret("HF_TOKEN"))

# hyperparameters and config
SERPER_API_KEY = user_secrets.get_secret("SERPER_API_KEY")  
GEMINI_API_KEY = user_secrets.get_secret("GEMINI_API_KEY")  
DATA_PATH = "..."
#Column names in your MedMCQA CSV-
QUESTION_COL = "question"
OPTION_COLS = ["opa", "opb", "opc", "opd"]  
ANSWER_COL = "cop"                         
LOCAL_MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
DEVICE_MAP = "auto"
GEMINI_MODEL_NAME = "gemini-2.5-flash-lite"
MAX_WEB_RESULTS = 5         
WEB_TIMEOUT = 10
MAX_QUESTIONS_EVAL = 100      
PRINT_FIRST_N_DEBUG = 3     
SLEEP_BETWEEN_GEMINI_CALLS = 5.0  
GEMINI_MAX_RETRIES = 3           

PROMPT_MODE = "instruction"         
VARIANTS = ["vanilla", "our_method"]  # vanilla = Gemini only; our_method = SLM+web+hints+Gemini




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


#helper functions
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


def call_gemini_with_cooldown(prompt, max_retries=GEMINI_MAX_RETRIES):
    """
    Wrapper around gemini_model.generate_content with:
      - base sleep between calls (SLEEP_BETWEEN_GEMINI_CALLS)
      - retry on 429 / transient errors using server-provided retry delay
    """
    if gemini_model is None:
        raise RuntimeError("Gemini model not initialised.")

    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = gemini_model.generate_content(prompt)
            time.sleep(SLEEP_BETWEEN_GEMINI_CALLS)
            return resp
        except Exception as e:
            last_exc = e
            msg = str(e)

            wait_s = None
            m1 = re.search(r"retry in ([0-9.]+)s", msg, flags=re.I)
            if m1:
                wait_s = float(m1.group(1))
            else:
                m2 = re.search(r"seconds:\s*([0-9]+)", msg, flags=re.I)
                if m2:
                    wait_s = float(m2.group(1))
            if wait_s is None:
                wait_s = 60.0

            print(f"[Gemini rate/HTTP error] Attempt {attempt}/{max_retries} → sleeping {wait_s:.1f}s...")
            time.sleep(wait_s)

    raise RuntimeError(f"Gemini generate_content failed after {max_retries} retries.") from last_exc


def extract_choice(out_text: str) -> str:
    """
    Robustly extract a single choice A/B/C/D from model output.
    Prefers a standalone letter near the end; falls back to first match.
    """
    if not out_text:
        return "A"
    t = out_text.upper()

    # Prefer a standalone A/B/C/D near the end of the string
    m = re.search(r"\b([ABCD])\b(?=[^ABCD]*$)", t)
    if m:
        return m.group(1)

    # Fallback: first A/B/C/D anywhere
    m = re.search(r"[ABCD]", t)
    if m:
        return m.group(0)

    return "A"

#SLM decomposes queries into WHY/WHAT/HOW
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



# SLM builds hints 
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

# Instruction Gemini prompt for MedMCQA

def _build_instruction_prompt_med(question, options_dict, facts):
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
            "Optional factual hints:\n"
            "- (none provided; rely on your own medical knowledge.)\n"
        )

    text += "\nAnswer (ONLY one letter A, B, C, or D):"
    return text


def gemini_mcq_answer(question, options_dict, facts, prompt_mode="instruction"):
    """
    Instruction-only MCQ answering:
      - Gemini sees question + options + OPTIONAL SLM hints and outputs A/B/C/D.
    """
    if gemini_model is None:
        return "A"

    text = _build_instruction_prompt_med(question, options_dict, facts)

    try:
        resp = call_gemini_with_cooldown(text)
        out_text = (resp.text or "").strip()
    except Exception as e:
        print("Gemini error while answering MedMCQA MCQ after retries:", e)
        out_text = ""

    return extract_choice(out_text)


# Mapping helper

def normalize_gold_label(raw):
    """
    MedMCQA 'cop' is often 0/1/2/3 (index of correct option).
    This converts various encodings (0-3, 1-4, A-D) into 'A'..'D'.
    """
    s = str(raw).strip()

    # Already a letter?
    if s.upper() in ["A", "B", "C", "D"]:
        return s.upper()

    # 0-3 index
    if s in ["0", "1", "2", "3"]:
        mapping = {"0": "A", "1": "B", "2": "C", "3": "D"}
        return mapping[s]

    # 1-4 index
    if s in ["1", "2", "3", "4"]:
        mapping = {"1": "A", "2": "B", "3": "C", "4": "D"}
        return mapping[s]

    return s.upper()


#Question pipeline

def predict_mcq_for_row(row, variant="our_method",
                        prompt_mode="instruction", verbose=False):
    """
    Pipeline for a single MedMCQA question:

      variant = "vanilla":
          Gemini only (no SLM, no web, no hints).
      variant = "our_method":
          1) Use SLM to decompose question into WHY / WHAT / HOW sub-queries.
          2) For each sub-query, retrieve 2 web snippets (total ~6 snippets).
          3) Tag snippets with the sub-query type and aggregate them.
          4) SLM builds structured hints from all tagged snippets.
          5) Gemini answers using question + options + OPTIONAL SLM hints.
    """
    question = str(row[QUESTION_COL])
    options = {
        "A": str(row[OPTION_COLS[0]]),
        "B": str(row[OPTION_COLS[1]]),
        "C": str(row[OPTION_COLS[2]]),
        "D": str(row[OPTION_COLS[3]]),
    }

    sub_queries = {}
    tagged_snippets = []
    facts = []

    if variant == "our_method":
        # 1) Decompose question into WHY/WHAT/HOW
        sub_queries = slm_decompose_queries(question)

        # 2) Retrieve web snippets for each sub-query (2 per query)
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

    # 4) Gemini answers using OPTIONAL SLM hints (for our_method) or no hints (for vanilla)
    ans_llm = gemini_mcq_answer(question, options, facts, prompt_mode=prompt_mode)
    final_pred = ans_llm

    if verbose:
        print("\n" + "=" * 80)
        print(f"Variant: {variant} | Prompt mode: {prompt_mode}")
        print(f"Question: {question}\n")
        print("Options:")
        for lab in ["A", "B", "C", "D"]:
            print(f"{lab}. {options[lab]}")
        if variant == "our_method":
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


# Final run

df = pd.read_csv(DATA_PATH)

print("Columns in MedMCQA csv:", list(df.columns))

total = min(MAX_QUESTIONS_EVAL, len(df))
rows = []

for variant in VARIANTS:
    print("\n" + "#" * 80)
    print(f"# Evaluating MedMCQA-100 — variant={variant}, prompt_mode={PROMPT_MODE}")
    print("#" * 80)

    predictions = []
    gold_labels = []
    correct_flags = []

    llm_answers = []
    all_sub_queries = []
    all_facts = []

    for idx in tqdm(range(total),
                    desc=f"MedMCQA-100 ({variant}, {PROMPT_MODE})"):
        row = df.iloc[idx]
        gold_raw = row[ANSWER_COL]
        gold = normalize_gold_label(gold_raw)
        gold_labels.append(gold)

        verbose = idx < PRINT_FIRST_N_DEBUG

        pred, ans_llm, sub_queries, facts = predict_mcq_for_row(
            row,
            variant=variant,
            prompt_mode=PROMPT_MODE,
            verbose=verbose
        )

        predictions.append(pred)
        llm_answers.append(ans_llm)
        all_sub_queries.append(sub_queries)
        all_facts.append(facts)

        is_corr = int(pred == gold)
        correct_flags.append(is_corr)

        if not verbose:
            print(f"[{variant}] Q{idx+1}: gold={gold} (raw={gold_raw}), pred={pred}, LLM={ans_llm}")

        rows.append({
            "q_index": idx,
            "variant": variant,
            "prompt_mode": PROMPT_MODE,
            "question": row[QUESTION_COL],
            "A": row[OPTION_COLS[0]],
            "B": row[OPTION_COLS[1]],
            "C": row[OPTION_COLS[2]],
            "D": row[OPTION_COLS[3]],
            "gold_raw": gold_raw,
            "gold_normalized": gold,
            "pred": pred,
            "correct": is_corr,
            "llm_answer_raw": ans_llm,
            "sub_queries": json.dumps(sub_queries),
            "facts": json.dumps(facts),
        })

    correct = sum(correct_flags)
    accuracy = correct / total
    print(f"\nFinished {total} MedMCQA questions for variant={variant}.")
    print(f"Pipeline accuracy (variant={variant}, mode={PROMPT_MODE}): {accuracy:.3f}")


# Save results

eval_df = pd.DataFrame(rows)
save_path = f"medmcqa_100_vanilla_vs_ourmethod_{PROMPT_MODE}_results.csv"
eval_df.to_csv(save_path, index=False)
print(f"\nSaved MedMCQA results (both variants) to: {save_path}")
