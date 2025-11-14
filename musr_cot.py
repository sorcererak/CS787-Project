import os, re, json, time, random, requests
import pandas as pd
from tqdm.auto import tqdm
import google.generativeai as genai
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login
from kaggle_secrets import UserSecretsClient
from datasets import load_dataset

user_secrets = UserSecretsClient()
HF_TOKEN = user_secrets.get_secret("HF_TOKEN")
SERPER_API_KEY = user_secrets.get_secret("SERPER_API_KEY")   
GEMINI_API_KEY = user_secrets.get_secret("GEMINI_API_KEY")   

login(token=HF_TOKEN)

model_name = "..."  
DEVICE_MAP = "auto"
llm_name = "..."

NUM_QUESTIONS = 100                 
SAMPLE_RANDOM = True                
PRINT_FIRST_N_DEBUG = 3
BASE_SLEEP_BETWEEN_CALLS = 5.0      
WEB_TIMEOUT = 10
MAX_WEB_RESULTS = 5

LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")  

gemini_model = None
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel(llm_name)
    print("Gemini model initialised.")
else:
    print("WARNING: GEMINI_API_KEY not set, Gemini LLM disabled.")

print("Loading local SLM (this can take a bit)...")
bnb_config = BitsAndBytesConfig(load_in_4bit=True)
slm_tokenizer = AutoTokenizer.from_pretrained(model_name)
slm_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map=DEVICE_MAP,
    quantization_config=bnb_config
)
print("Local SLM loaded.")

gemini_token_stats = {"prompt": 0, "output": 0, "total": 0, "calls": 0}

def _accumulate_usage(resp):
    usage = getattr(resp, "usage_metadata", None)
    if usage is not None:
        pt = getattr(usage, "prompt_token_count", 0)
        ct = getattr(usage, "candidates_token_count", 0)
        tt = getattr(usage, "total_token_count", pt + ct)
        gemini_token_stats["prompt"] += pt
        gemini_token_stats["output"] += ct
        gemini_token_stats["total"] += tt
    gemini_token_stats["calls"] += 1

def call_gemini_with_cooldown(prompt, max_retries=3):
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = gemini_model.generate_content(prompt)
            _accumulate_usage(resp)
            time.sleep(BASE_SLEEP_BETWEEN_CALLS)
            return resp
        except Exception as e:
            last_exc = e
            msg = str(e)
            wait_s = None
            m1 = re.search(r"retry in ([0-9.]+)s", msg)
            if m1: wait_s = float(m1.group(1))
            else:
                m2 = re.search(r"seconds:\s*([0-9]+)", msg)
                if m2: wait_s = float(m2.group(1))
            if wait_s is None: wait_s = 60.0
            print(f"[Rate limit] Attempt {attempt}/{max_retries} → sleeping {wait_s:.1f}s...")
            time.sleep(wait_s)
    raise RuntimeError("Gemini generate_content failed after retries.") from last_exc

def slm_generate(prompt, max_new_tokens=128):
    inputs = slm_tokenizer(prompt, return_tensors="pt").to(slm_model.device)
    out_ids = slm_model.generate(**inputs, max_new_tokens=max_new_tokens)
    text = slm_tokenizer.decode(out_ids[0], skip_special_tokens=True)
    return text.strip()

def web_search(query, num_results=2):
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

def slm_decompose_queries(narrative, question):
    prompt = (
        "You are a query decomposition assistant for narrative reasoning MCQs (MuSR-like).\n"
        "Rewrite the problem into exactly three concise sub-queries:\n"
        "1) WHY (reason/cause/explanation about the key event or answer)\n"
        "2) WHAT (key facts, entities, or clues from the narrative)\n"
        "3) HOW (mechanism/process/logic linking clues to the answer)\n\n"
        "Return EXACTLY:\nWHY: <why>\nWHAT: <what>\nHOW: <how>\n\n"
        f"NARRATIVE:\n{narrative}\n\nQUESTION:\n{question}\n\nDecomposed queries:"
    )
    text = slm_generate(prompt, max_new_tokens=192)
    why = what = how = question
    m_why = re.search(r"WHY:\s*(.+)", text, flags=re.IGNORECASE)
    m_what = re.search(r"WHAT:\s*(.+)", text, flags=re.IGNORECASE)
    m_how = re.search(r"HOW:\s*(.+)", text, flags=re.IGNORECASE)
    if m_why:  why  = m_why.group(1).strip()
    if m_what: what = m_what.group(1).strip()
    if m_how:  how  = m_how.group(1).strip()
    return {"WHY": why, "WHAT": what, "HOW": how}

def slm_build_hints(narrative, question, tagged_snippets):
    prompt = (
        "You are helping with a narrative reasoning MCQ (MuSR-like).\n"
        "Given the narrative, the question, and web snippets tagged WHY/WHAT/HOW,\n"
        "extract 4–8 short factual bullet points (salient clues, relations, constraints).\n"
        "Do NOT answer the question. Each bullet = one simple statement.\n\n"
        f"NARRATIVE:\n{narrative}\n\nQUESTION:\n{question}\n\nWEB SNIPPETS:\n"
    )
    for i, s in enumerate(tagged_snippets, 1):
        prompt += f"[{i}] {s['snippet']}\n"
    prompt += "\nNow write 4–8 bullet points, each starting with '- ' only:\n"
    text = slm_generate(prompt, max_new_tokens=256)
    facts = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("-"):
            fact = line.lstrip("-").strip()
            if fact: facts.append(fact)
    if not facts:
        prompt2 = (
            "List 4–8 short bullet point facts from the narrative relevant to the question.\n"
            "Do not answer the question. Bullets only, begin with '- '.\n\n"
            f"NARRATIVE:\n{narrative}\n\nQUESTION:\n{question}\n"
        )
        text2 = slm_generate(prompt2, max_new_tokens=192)
        for line in text2.splitlines():
            line = line.strip()
            if line.startswith("-"):
                fact = line.lstrip("-").strip()
                if fact: facts.append(fact)
    return facts

def gemini_mcq_answer_with_hints(narrative, question, options_list, facts):
    k = len(options_list)
    letters = LETTERS[:k]
    options_block = "\n".join(f"{letters[i]}. {options_list[i]}" for i in range(k))
    allowed_str = ", ".join(letters)

    text = (
        "You are a strong narrative reasoning model for MuSR.\n"
        "First, think carefully and silently through the reasoning process.\n"
        "Use the narrative, question, options, and optional hints.\n"
        "Do NOT reveal any reasoning steps.\n"
        "After thinking internally, output ONLY the final answer letter.\n\n"
        f"NARRATIVE:\n{narrative}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"OPTIONS:\n{options_block}\n\n"
    )

    if facts:
        text += "Optional hints:\n" + "".join(f"- {f}\n" for f in facts)

    text += (
        f"\nFinal Answer: Output ONLY ONE capital letter from {{{allowed_str}}}, "
        "with no explanation."
    )

    resp = call_gemini_with_cooldown(text)
    out = (resp.text or "").strip()
    m = re.search(r"[A-Z]", out)
    if m and m.group(0) in letters:
        return m.group(0)
    return letters[0]


def gemini_mcq_answer_vanilla(narrative, question, options_list):
    k = len(options_list)
    letters = LETTERS[:k]
    options_block = "\n".join(f"{letters[i]}. {options_list[i]}" for i in range(k))
    allowed_str = ", ".join(letters)

    prompt = (
        "You are answering a narrative reasoning multiple-choice question (MuSR-like).\n"
        "Think silently. Return ONLY ONE capital letter from the allowed set.\n\n"
        f"NARRATIVE:\n{narrative}\n\nQUESTION:\n{question}\n\nOPTIONS:\n{options_block}\n\n"
        f"Answer (ONE letter in {{{allowed_str}}}):"
    )
    resp = call_gemini_with_cooldown(prompt)
    out = (resp.text or "").strip()
    m = re.search(r"[A-Z]", out)
    if m and m.group(0) in letters:
        return m.group(0)
    return letters[0]

ds = load_dataset("TAUR-Lab/MuSR")

splits = []
for split_name in ["murder_mysteries", "object_placements", "team_allocation"]:
    if split_name in ds:
        for i in range(len(ds[split_name])):
            splits.append((split_name, i))

if SAMPLE_RANDOM:
    random.seed(42)
    random.shuffle(splits)

splits = splits[:NUM_QUESTIONS]
print(f"Sampling {len(splits)} items across splits:", pd.Series([s for s,_ in splits]).value_counts().to_dict())

import json, ast
rows = []

def safe_parse_choices(raw_choices):
    """Ensure `choices` is always a Python list of strings."""
    if isinstance(raw_choices, list):
        return raw_choices
    if isinstance(raw_choices, str):
        try:
            return json.loads(raw_choices)  
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(raw_choices)  
            except Exception as e:
                print("Could not parse choices:", raw_choices, "| Error:", e)
                return []
    return []

print("Running VANILLA Gemini on MuSR:")
vanilla_correct = 0
for j, (split_name, idx0) in tqdm(list(enumerate(splits)), desc="Vanilla"):
    ex = ds[split_name][idx0]
    narrative = ex["narrative"].strip()
    question = ex["question"].strip()
    
    choices = safe_parse_choices(ex["choices"])
    k = len(choices)
    assert k >= 2, f"MuSR item has < 2 choices or malformed: {ex['choices']}"
    
    gold_idx = int(ex["answer_index"])
    gold_letter = LETTERS[gold_idx]

    pred = gemini_mcq_answer_vanilla(narrative, question, choices)
    ok = int(pred == gold_letter)
    vanilla_correct += ok

    if j < PRINT_FIRST_N_DEBUG:
        print("\n" + "="*60)
        print(f"[{split_name}:{idx0}] Q: {question}")
        print(f"Narrative: {narrative[:300]}{'...' if len(narrative)>300 else ''}")
        for li, c in enumerate(choices):
            print(f"{LETTERS[li]}. {c}")
        print(f"Vanilla: {pred} | Gold: {gold_letter} | Correct: {bool(ok)}")

    rows.append({
        "id": f"{split_name}:{idx0}",
        "split": split_name,
        "variant": "vanilla",
        "narrative": narrative,
        "question": question,
        "options": json.dumps(choices, ensure_ascii=False),
        "k": k,
        "pred": pred,
        "gold": gold_letter,
        "correct": ok
    })

vanilla_acc = vanilla_correct / len(splits)
print(f"\nVanilla Gemini accuracy (MuSR, n={len(splits)}): {vanilla_acc:.3f}")

print("Running OUR METHOD (SLM decomp + web hints + Gemini) on MuSR:")
gemini_token_stats = {"prompt": 0, "output": 0, "total": 0, "calls": 0}

our_correct = 0
for j, (split_name, idx0) in tqdm(list(enumerate(splits)), desc="OurMethod"):
    ex = ds[split_name][idx0]
    narrative = ex["narrative"].strip()
    question = ex["question"].strip()

    choices = safe_parse_choices(ex["choices"])
    k = len(choices)
    assert k >= 2, f"MuSR item has < 2 choices or malformed: {ex['choices']}"
    
    gold_idx = int(ex["answer_index"])
    gold_letter = LETTERS[gold_idx]

    sub_queries = slm_decompose_queries(narrative, question)

    tagged_snippets = []
    for tag in ["WHY", "WHAT", "HOW"]:
        subq = sub_queries[tag]
        for s in web_search(subq, num_results=2):
            tagged_snippets.append({"snippet": f"[{tag}] {s['snippet']}", "url": s["url"]})

    facts = slm_build_hints(narrative, question, tagged_snippets)

    pred = gemini_mcq_answer_with_hints(narrative, question, choices, facts)
    ok = int(pred == gold_letter)
    our_correct += ok

    if j < PRINT_FIRST_N_DEBUG:
        print("\n" + "="*60)
        print(f"[{split_name}:{idx0}] Q: {question}")
        print(f"Sub-queries: {sub_queries}")
        print("Sample hints:", facts[:5])
        print(f"OurMethod: {pred} | Gold: {gold_letter} | Correct: {bool(ok)}")

    rows.append({
        "id": f"{split_name}:{idx0}",
        "split": split_name,
        "variant": "our_method",
        "narrative": narrative,
        "question": question,
        "options": json.dumps(choices, ensure_ascii=False),
        "k": k,
        "pred": pred,
        "gold": gold_letter,
        "correct": ok,
        "sub_queries": json.dumps(sub_queries, ensure_ascii=False),
        "facts": json.dumps(facts[:8], ensure_ascii=False),
    })

our_acc = our_correct / len(splits)
print(f"\nOur method accuracy (MuSR, n={len(splits)}): {our_acc:.3f}")

df_out = pd.DataFrame(rows)
out_path = "musr_100_eval_vanilla_vs_ourmethod.csv"
df_out.to_csv(out_path, index=False)
print("Saved:", out_path)

print("\nGemini token usage for OurMethod:")
print(f"  Calls         : {gemini_token_stats['calls']}")
print(f"  Prompt tokens : {gemini_token_stats['prompt']}")
print(f"  Output tokens : {gemini_token_stats['output']}")
print(f"  Total tokens  : {gemini_token_stats['total']}")
if gemini_token_stats["calls"] > 0:
    print(f"  Avg tokens / call    : {gemini_token_stats['total']/gemini_token_stats['calls']:.1f}")
print(f"  Avg tokens / question: {gemini_token_stats['total']/max(1,len(splits)):.1f}")

print("\nSummary:")
print(f"  Vanilla accuracy : {vanilla_acc:.3f}")
print(f"  OurMethod accuracy: {our_acc:.3f}")