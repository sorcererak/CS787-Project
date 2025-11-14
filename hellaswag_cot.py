import os, re, json, time, requests, random
import pandas as pd
from tqdm.auto import tqdm
import google.generativeai as genai
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login
from kaggle_secrets import UserSecretsClient
from datasets import load_dataset

user_secrets = UserSecretsClient()
HF_TOKEN = user_secrets.get_secret("HF_TOKEN")
SERPER_API_KEY = user_secrets.get_secret("SERPER_API_KEY")   # optional
GEMINI_API_KEY = user_secrets.get_secret("GEMINI_API_KEY")   # required
login(token=HF_TOKEN)

model_name = ".."
device = "auto"
llm_name = "..."

NUM_QUESTIONS = 100
random_sample = True
sleep_time = 5.0
web_timeout = 10
web_results = 5

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel(llm_name)
print("Gemini model initialised.")
print("Loading SLM")

bnb_config = BitsAndBytesConfig(load_in_4bit=True)
slm_tokenizer = AutoTokenizer.from_pretrained(model_name
slm_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map=device,
    quantization_config=bnb_config
)

print("Local SLM loaded.")

def call_gemini_with_cooldown(prompt, max_retries=3):
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = gemini_model.generate_content(prompt)
            _accumulate_usage(resp)
            time.sleep(sleep_time)
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

def slm_generate(prompt, max_new_tokens=96):
    inputs = slm_tokenizer(prompt, return_tensors="pt").to(slm_model.device)
    out_ids = slm_model.generate(**inputs, max_new_tokens=max_new_tokens)
    text = slm_tokenizer.decode(out_ids[0], skip_special_tokens=True)
    return text.strip()

def web_search(query, num_results=2):
    num_results = min(num_results, web_results)
    results = []
    if SERPER_API_KEY:
        url = "httpsa://google.serper.dev/search"
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
        "You are a query decomposition assistant for commonsense/story continuation MCQs.\n"
        "Rewrite the question into exactly three concise sub-queries:\n"
        "1) WHY (reason/cause/explanation)\n"
        "2) WHAT (key entities/events)\n"
        "3) HOW (likely next action/mechanism)\n\n"
        "Return EXACTLY:\n"
        "WHY: <why>\nWHAT: <what>\nHOW: <how>\n\n"
        f"Original question: {question}\n\nDecomposed queries:"
    )
    text = slm_generate(prompt, max_new_tokens=128)
    why = what = how = question
    m_why = re.search(r"WHY:\s*(.+)", text, flags=re.IGNORECASE)
    m_what = re.search(r"WHAT:\s*(.+)", text, flags=re.IGNORECASE)
    m_how = re.search(r"HOW:\s*(.+)", text, flags=re.IGNORECASE)
    if m_why:  why  = m_why.group(1).strip()
    if m_what: what = m_what.group(1).strip()
    if m_how:  how  = m_how.group(1).strip()
    return {"WHY": why, "WHAT": what, "HOW": how}

def slm_build_hints(question, tagged_snippets):
    prompt = (
        "You are helping with a commonsense/story continuation MCQ (HellaSwag-like).\n"
        "Given the question and web snippets tagged by WHY/WHAT/HOW, extract 4–8 short\n"
        "factual/contextual bullet points that could help judge the most plausible continuation.\n"
        "Do NOT answer the question. Each bullet = a single simple statement.\n\n"
        f"Question: {question}\n\nWeb snippets:\n"
    )
    for i, s in enumerate(tagged_snippets, 1):
        prompt += f"[{i}] {s['snippet']}\n"
    prompt += "\nNow write 4–8 bullet points, each starting with '- ' only:\n"
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
        "You are answering a commonsense/story continuation multiple-choice question "
        "(HellaSwag-like). Choose the most plausible continuation.\n\n"
        "You will also see OPTIONAL helper hints produced by a smaller model from web search.\n"
        "Use them only if they meaningfully support your reasoning; otherwise ignore them.\n\n"
        "Follow this reasoning strategy:\n"
        "  1. Read the context carefully and understand the scenario.\n"
        "  2. Think step-by-step about how real people, objects, or events typically behave.\n"
        "  3. Use commonsense, physical plausibility, and narrative coherence.\n"
        "  4. Compare each option and eliminate implausible story continuations.\n"
        "  5. Choose the option that best fits what naturally happens next.\n\n"
        "Important formatting instructions:\n"
        "  i) Show your reasoning explicitly.\n"
        " ii) At the end, output the answer using this format:\n"
        "        Final Answer: X\n"
        "     where X is one of A, B, C, or D.\n\n"
        f"Question (context): {question}\n\n"
        "Options:\n"
        f"A. {options_dict['A']}\n"
        f"B. {options_dict['B']}\n"
        f"C. {options_dict['C']}\n"
        f"D. {options_dict['D']}\n\n"
    )

    if facts:
        text += "Optional hints:\n" + "".join(f"- {f}\n" for f in facts)
    text += "\nAnswer:"
    resp = call_gemini_with_cooldown(text)
    out = (resp.text or "").strip()
    m = re.search(r"[ABCD]", out)
    return m.group(0) if m else "A"

def gemini_mcq_answer_vanilla(question, options_dict):
    prompt = (
        "You are answering a commonsense/story continuation multiple-choice question "
        "(HellaSwag-like). Choose the most plausible continuation.\n"
        "Think silently. Return ONLY a single capital letter: A, B, C, or D.\n\n"
        f"Context: {question}\n\n"
        f"A. {options_dict['A']}\n"
        f"B. {options_dict['B']}\n"
        f"C. {options_dict['C']}\n"
        f"D. {options_dict['D']}\n\n"
        "Answer:"
    )
    resp = call_gemini_with_cooldown(prompt)
    out = (resp.text or "").strip()
    m = re.search(r"[ABCD]", out)
    return m.group(0) if m else "A"

ds = load_dataset("Rowan/hellaswag")
val = ds["validation"]
idxs = list(range(len(val)))

if random_sample:
    random.seed(42)
    random.shuffle(idxs)
idxs = idxs[:NUM_QUESTIONS]
sub = val.select(idxs)

def idx_to_letter(i): 
    return ["A","B","C","D"][int(i)]

def letter_to_idx(L): 
    return {"A":0,"B":1,"C":2,"D":3}[L]

results_rows = []

print("\nRunning VANILLA Gemini on HellaSwag")

vanilla_correct = 0

for i in tqdm(range(len(sub)), desc="Vanilla"):
    ex = sub[i]
    context = ex["ctx"].strip()
    endings = [e.strip() for e in ex["endings"]]
    gold_letter = idx_to_letter(ex["label"])

    options = {"A": endings[0], "B": endings[1], "C": endings[2], "D": endings[3]}
    pred = gemini_mcq_answer_vanilla(context, options)
    ok = int(pred == gold_letter)
    vanilla_correct += ok
    
    print(f"Vanilla: {pred} | Gold: {gold_letter} | Correct: {bool(ok)}")
    results_rows.append({
        "id": int(idxs[i]),
        "variant": "vanilla",
        "context": context,
        "A": options["A"], "B": options["B"], "C": options["C"], "D": options["D"],
        "pred": pred, "gold": gold_letter, "correct": ok
    })

vanilla_acc = vanilla_correct/len(sub)
print(f"\nVanilla Gemini accuracy (HellaSwag, n={len(sub)}): {vanilla_acc:.3f}")

print("\n=== Running our method on HellaSwag (validation) ===")
gemini_token_stats = {"prompt": 0, "output": 0, "total": 0, "calls": 0}
our_correct = 0
for i in tqdm(range(len(sub)), desc="OurMethod"):
    ex = sub[i]
    context = ex["ctx"].strip()
    endings = [e.strip() for e in ex["endings"]]
    gold_letter = idx_to_letter(ex["label"])
    options = {"A": endings[0], "B": endings[1], "C": endings[2], "D": endings[3]}
    question = f"Choose the most plausible continuation.\nContext: {context}"
    sub_queries = slm_decompose_queries(question)
    tagged_snippets = []
    for tag in ["WHY", "WHAT", "HOW"]:
        subq = sub_queries[tag]
        for s in web_search(subq, num_results=2):
            tagged_snippets.append({"snippet": f"[{tag}] {s['snippet']}", "url": s["url"]})
    facts = slm_build_hints(question, tagged_snippets)
    pred = gemini_mcq_answer_with_hints(question, options, facts)
    ok = int(pred == gold_letter)
    our_correct += ok
    print(f"OurMethod: {pred} | Gold: {gold_letter} | Correct: {bool(ok)}")

    results_rows.append({
        "id": int(idxs[i]),
        "variant": "our_method",
        "context": context,
        "A": options["A"], "B": options["B"], "C": options["C"], "D": options["D"],
        "pred": pred, "gold": gold_letter, "correct": ok,
        "sub_queries": json.dumps(sub_queries),
        "facts": json.dumps(facts[:8]),
    })

our_acc = our_correct/len(sub)
print(f"\nOur method accuracy (HellaSwag, n={len(sub)}): {our_acc:.3f}")

df_out = pd.DataFrame(results_rows)
df_out.to_csv("hellaswag_instruct.csv", index=False)

print(f"Vanilla accuracy : {vanilla_acc:.3f}")
print(f"OurMethod accuracy: {our_acc:.3f}")