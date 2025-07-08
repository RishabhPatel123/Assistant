import torch, requests, re, time, os, json ,random
from langdetect import detect
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from peft import PeftModel

# === Memory Files ===
chat_history = []
HISTORY_FILE = "chat_history.txt"
MEMORY_FILE = "user_memory.json"

# === Load Tokenizer and LoRA Adapter ===
tokenizer = GPT2Tokenizer.from_pretrained(os.path.join("data", "baalak_adapter"))
tokenizer.pad_token = tokenizer.eos_token

base_model = GPT2LMHeadModel.from_pretrained("gpt2-medium")
model = PeftModel.from_pretrained(base_model, os.path.join("data", "baalak_adapter")).to(
    torch.device("cuda" if torch.cuda.is_available() else "cpu")
).eval()
device = model.device

# === Static Priming ===
STATIC_PRIMING = (
    "The following is a friendly, emotional, romantic chatbot named Baalak.\n"
    "He talks like a human, with feelings and fun.\n"
    "Baalak always responds naturally, short ,accurate and warmly to any input.\n"
)

# === Load & Save Memory ===
def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_memory(data):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

user_memory = load_memory()

def insert_memory_context():
    name = user_memory.get("name")
    return f"User's name is {name}.\n" if name else ""

def update_memory(prompt):
    name_match = re.search(r"(?:my name is|call me)\s+(\w+)", prompt.lower())
    if name_match:
        user_memory["name"] = name_match.group(1).capitalize()
        save_memory(user_memory)

# === Typing Effect ===
def print_typing(text, delay=0.02):
    for char in text:
        print(char, end='', flush=True)
        time.sleep(delay)
    print()

# === Mood Detection ===
def get_mood(prompt):
    p = prompt.lower()
    if any(w in p for w in ["love", "miss", "sweetheart", "romantic", "pyaar", "hug"]): return "romantic"
    if any(w in p for w in ["sad", "cry", "hurt", "breakup", "dukhi"]): return "emotional"
    if any(w in p for w in ["joke", "laugh", "funny", "meme"]): return "funny"
    if any(w in p for w in ["angry", "hate", "stupid", "idiot"]): return "attitude"
    return "default"

# === Intent Classifier ===
def classify_intent(prompt):
    p = prompt.lower()
    if re.search(r"(what|who|where|when|how|कौन|क्या)", p): return "factual"
    if "love" in p: return "romantic"
    if "sad" in p: return "emotional"
    if "joke" in p: return "funny"
    return "default"

# === Wikipedia Lookup ===
def wiki_summary(query):
    try: lang = detect(query)
    except: lang = "en"
    try:
        params = {"action": "query", "format": "json", "prop": "extracts", "exintro": True, "explaintext": True, "titles": query, "redirects": 1}
        resp = requests.get(f"https://{lang}.wikipedia.org/w/api.php", params=params, timeout=5)
        page = next(iter(resp.json()["query"]["pages"].values()))
        return page.get("extract"), lang
    except: return None, lang

# === GPT2 Response Generator ===
def gpt2_reply(prompt):
    update_memory(prompt)
    ctx = "\n".join(chat_history[-4:])
    mood = get_mood(prompt)
    input_text = STATIC_PRIMING + insert_memory_context() + f"<Mood: {mood}>\n" + ctx + f"\nYou: {prompt}\nBaalak:"

    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512, padding="max_length").to(device)

    with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
        output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=60,
            temperature=0.6,
            top_k=40,
            top_p=0.8,
            do_sample=True,
            repetition_penalty=1.15,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    reply = re.split(r"(?<=[.!?])\s+", decoded)[0]
    reply = re.sub(r"[^\w\s.,!?ँ-ॿa-zA-Z0-9]", "", reply)
    return reply.strip().capitalize()

# === Final Reply Wrapper ===
def generate_response(prompt):
    if classify_intent(prompt) == "factual":
        summary, lang = wiki_summary(prompt)
        if summary:
            first_line = summary.split("\n")[0][:300]
            emoji = "❤️" if lang == "hi" else "📘"
            return f"Baalak: ({lang.upper()}) {first_line.strip()} {emoji}"
    return f"Baalak: {gpt2_reply(prompt)}"

# === Chat Loop ===
if __name__ == "__main__":
    print(f"\n🚀 Baalak running on: {device}\n🧠 Baalak is ready. Type 'quit' to exit.\n")
    while True:
        user = input("You : ").strip()
        if user.lower() in ["quit", "exit"]:
            print("Baalak: Bye yaar, take care ❤️")
            with open(HISTORY_FILE, "a", encoding="utf-8") as chat:
                chat.write("\n".join(chat_history[-20:]))
            break
        reply = generate_response(user)
        print_typing(reply)
        chat_history.extend([f"You: {user}", reply])
        time.sleep(1)
