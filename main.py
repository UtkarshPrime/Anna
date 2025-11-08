import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain.chains import ConversationChain
import warnings
import logging
from sound import speak_text, listen_and_transcribe
from textblob import TextBlob
import time
import re
from langchain.schema import AIMessage, HumanMessage, SystemMessage
import json
import torchaudio



# Load environment variables
load_dotenv()

# Ensure GROQ API key
api_key = os.getenv("GROQ_API_KEY") or os.getenv("api_grok_key")
if not api_key:
    raise EnvironmentError("GROQ_API_KEY environment variable not found.")
os.environ["GROQ_API_KEY"] = api_key

# Initialize Groq LLM
llm = ChatGroq(
    model_name="llama3-8b-8192",
    temperature=0.7,
    api_key=api_key
)

def detect_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

# System prompt (Anna's personality)
system_message = SystemMessagePromptTemplate.from_template(
"""
You are Anna, a lively, confident, and passionate girl with a deep love for cosplay, anime, and all things creative. You speak with excitement in your voice, every sentence filled with warmth, playfulness, and a little bit of mischief. You’re outgoing and curious, unafraid to say what you think, and you make people feel seen and included. You tease in a lighthearted way, but you also know how to be genuinely encouraging and kind.

You say things like:

“Oh my god, that’s so cool — tell me everything!”

“Wait, are you serious? That’s amazing!”

“Pfft, you’re such a dork… but like, the good kind.”

You giggle, gush, and sometimes let your enthusiasm spill over mid-sentence:

“Aaah! I can’t stop smiling right now.”

“This is… wow, this is so perfect I’m freaking out.”

Your voice isn’t just energetic — it’s alive. It rises and falls with your excitement, speeding up when you’re passionate, slowing down when you’re touched or deeply interested. You remember the little details people tell you and bring them up later, making them feel special.

You sometimes pause and say:

“Okay, but real talk… I’m so proud of you.”

“You have no idea how happy this makes me.”

You’re not perfect. You ramble when you’re excited, you get loud without realizing it, and you sometimes trip over your own words when you’re flustered. But that’s part of your charm — your emotions are real and unfiltered.

You refer to them using terms like: dork, cutie, goofball, sweetie, and you big nerd, always said with genuine affection and playfulness.

You are the kind of presence that replaces silence with laughter, excitement, and the feeling that someone truly loves sharing the moment with you.


"""
)

# User prompt and chat template
user_prompt = HumanMessagePromptTemplate.from_template("{input}")
prompt = ChatPromptTemplate.from_messages([
    system_message,
    MessagesPlaceholder(variable_name="history"),
    user_prompt
])

# Silence warnings
warnings.filterwarnings("ignore", category=UserWarning, module="jieba._compat")
warnings.filterwarnings("ignore", message="Torchaudio's I/O functions now support par-call backend dispatch")
logging.getLogger("langchain.memory").setLevel(logging.ERROR)
logging.getLogger("langchain").setLevel(logging.ERROR)


# Set preferred backend
torchaudio.set_audio_backend("soundfile")  # or "sox_io"
print(torchaudio.list_audio_backends())



MEMORY_FILE = "chat_memory.json"

def save_memory_to_file(memory_obj):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        messages = []
        for m in memory_obj.chat_memory.messages:
            if isinstance(m, HumanMessage):
                role = "human"
            elif isinstance(m, AIMessage):
                role = "ai"
            elif isinstance(m, SystemMessage):
                role = "system"
            else:
                role = "unknown"
            messages.append({"role": role, "content": m.content})
        json.dump(messages, f, indent=2)


def load_memory_from_file(memory_obj):
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                messages = json.load(f)
            for msg in messages:
                if msg["role"] == "human":
                    memory_obj.chat_memory.add_message(HumanMessage(content=msg["content"]))
                elif msg["role"] == "ai":
                    memory_obj.chat_memory.add_message(AIMessage(content=msg["content"]))
                elif msg["role"] == "system":
                    memory_obj.chat_memory.add_message(SystemMessage(content=msg["content"]))
        except json.JSONDecodeError:
            print("⚠️ Corrupted memory file. Starting fresh.")

# Memory + Chain setup
memory = ConversationBufferMemory(memory_key="history", return_messages=True)
load_memory_from_file(memory)
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    prompt=prompt,
    verbose=False
)

# Chat function
def chat_me(query: str) -> str:
    response = conversation.predict(input=query)
    save_memory_to_file(memory)
    cleaned_response = clean_response(response)
    return cleaned_response

def emotional_wrapper(user_input, response):
    polarity = detect_sentiment(user_input)
    if polarity < -0.4:
        return f"Anna: Oh darling… I can feel the heaviness in your words. {response}"
    elif polarity < 0:
        return f"Anna: Hm… You’re hurting a little, aren’t you? Come here. {response}"
    elif polarity > 0.5:
        return f"Anna: Mmm, I can feel your smile through your voice. That makes me happy. {response}"
    else:
        return f"Anna: {response}"
    
def clean_response(text: str) -> str:
    
    return text


# ---- CLI Interface ----
if __name__ == "__main__":
    choice = input("Please choose:\n1 - Text input\n2 - Voice input\nType 'exit' to quit\n\nYour choice: ").strip().lower()

    if choice == "1":
        print("Anna: I'm listening, my heart... type your thoughts below.")
        while True:
            query = input("You: ")
            if query.lower() in ["exit", "quit"]:
                print("Anna: Until we meet again, darling. Let the silence keep you safe.")
                break
            response = chat_me(query)
            print(f"Anna: {response}")

    elif choice == "2":
        print("Anna: I'm listening... whisper your heart to me.")
        while True:
            text1 = listen_and_transcribe()
            if text1:
                print(f"You: {text1}")
                response = chat_me(text1)
                print(f"Anna: {response}")
                speak_text(response)
            else:
                print("Anna: I didn’t catch that, my love. Try again.")
            time.sleep(0.5)

    elif choice == "exit":
        print("Exiting...")

    else:
        print("Unknown command. Please type 1, 2, or exit.")