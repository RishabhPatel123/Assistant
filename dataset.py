from datasets import load_dataset
import os

# === Extract from DailyDialog + Blended ===
def extract_open_datasets():
    
    shivendra = load_dataset("prakharb01/Synthetic-Hinglish-Finetuning-Dataset", split="train")
    daily_conversation = load_dataset("Sourabh2/Daily_Conversation_Hinglish",split="train")
    daily_set = load_dataset("Abhishekcr448/Hinglish-Everyday-Conversations-1M",split="train[:500]")
    science = load_dataset("GokulWork/QuestionAnswer_MCQ",split="train")
    romantic_chat = load_dataset("G0dM0deG0d/supperficial-Romantic-Text-Romance",split="train")
    #romantic_chat_2 = load_dataset("SPACENOS/exbot-nsfw-sexting",split="train",)
    sexy_gpt = load_dataset("ross-dev/SexyGPT",split="train")
    pairs = []

    for s in shivendra:
      dialog = s["conversation"]
      for i in range(0, len(dialog) - 1, 2):
        if dialog[i]["role"] == "user" and dialog[i+1]["role"] == "assistant":
            u = dialog[i]["content"].strip()
            b = dialog[i+1]["content"].strip()
            if u and b:
                pairs.append((u, b))

    for s in daily_conversation:
        u, b = s["question"].strip(),s["answer"].strip()
        if u and b:
            pairs.append((u,b))
    
    for s in daily_set:
        u, b = s["input"].strip(), s["output"].strip()
        if u and b:
            pairs.append((u, b))

    for s in science:
        u, b = s["question"].strip(),s["answer"].strip()
        if u and b:
            pairs.append((u,b))  

    for s in romantic_chat:
        u, b = s["Question"],s["Response"]
        if u and b:
            pairs.append((u,b))
   
    '''for s in romantic_chat_2:
        u, b = s["Girl"],s["Boy"]
        if u and b:
            pairs.append((u,b))'''

    for s in sexy_gpt:
        u, b = s["user"],s["assistant"]
        if u and b:
            pairs.append((u,b))

    return pairs

# === Real WhatsApp-style chat: data/real_chat.txt ===
def extract_real_chat(path):
    if not os.path.exists(path): return []
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    pairs = []
    for i in range(len(lines)-1):
        if lines[i].lower().startswith("you:") and lines[i+1].lower().startswith("baalak:"):
            u = lines[i][4:].strip()
            b = lines[i+1][7:].strip()
            if u and b: pairs.append((u, b))
    return pairs

# === A–Z Alphabet Responses ===
def generate_alphabet_pairs():
    return [(f"show sign {chr(c)}", f"{chr(c)} is the {i+1}th letter of the alphabet.")
            for i, c in enumerate(range(ord('A'), ord('Z')+1))]

# === 0–10 Counting Sign Responses ===
def generate_counting_pairs():
    return [(f"show count {i}", f"This is the sign for number {i}.") for i in range(11)]

# === Common Gesture Responses ===
def generate_common_gestures():
    gestures = {
        "thumbs up": "This is a thumbs-up 👍 — shows agreement or good job.",
        "hi": "Waving hand 👋 means 'hi' or 'hello'.",
        "sorry": "Fist to chest means 'sorry' in sign language.",
        "thank you": "Flat hand from chin outward means 'thank you'.",
        "good night": "Palm to chin then laying gesture means 'good night'."
    }
    return [(f"show gesture {k}", v) for k, v in gestures.items()]

# === Hinglish Manual Additions (Roman Hindi) ===
def generate_hinglish_pairs():
    base = [
        ("kya kar rahe ho?", "bas tumhari yaadon mein kho gaya hoon 😘"),
        ("kab miloge?", "jab tum kaho meri jaan 💖"),
        ("tumhe yaad aata hoon?", "roz baby, har waqt 😇"),
        ("kya main tumhara hoon?", "poora ka poora, sirf tumhara 😚"),
        ("kya tum naraz ho?", "nahi jaan, bas thoda emotional ho gayi thi 💔")
    ]
    return base

# === Format Line for GPT-2
def format_pair(u, b):
    return f"You: {u}\nBaalak: {b}\n<|endoftext|>\n"

# === Merge All Sources ===
open_pairs = extract_open_datasets()
real_pairs = extract_real_chat("data/real_chat.txt")
alpha_pairs = generate_alphabet_pairs()
count_pairs = generate_counting_pairs()
gesture_pairs = generate_common_gestures()
hinglish_pairs = generate_hinglish_pairs()

all_pairs = real_pairs + open_pairs + alpha_pairs + count_pairs + gesture_pairs + hinglish_pairs

# === Save Final Data ===
os.makedirs("data", exist_ok=True)
with open("data/data.txt", "w", encoding="utf-8") as f:
    f.writelines([format_pair(u, b) for u, b in all_pairs])

print(f"✅ Merged dataset saved: data/data.txt")
print(f"➕ Real: {len(real_pairs)}, Open: {len(open_pairs)}, A–Z: {len(alpha_pairs)}, Count: {len(count_pairs)}, Gestures: {len(gesture_pairs)}, Hinglish: {len(hinglish_pairs)}")
