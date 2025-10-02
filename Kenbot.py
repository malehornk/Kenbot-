from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_NAME = "facebook/blenderbot-400M-distill"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

history = []

def chat(prompt):
    history.append(f"User: {prompt}")
    inputs = tokenizer("\n".join(history[-5:]), return_tensors="pt")
    output = model.generate(**inputs, max_length=200, temperature=0.6, top_p=0.85, top_k=40, do_sample=True)
    reply = tokenizer.decode(output[0], skip_special_tokens=True)
    history.append(f"KenBot: {reply}")
    return reply

if __name__ == "__main__":
    print("KenBot v0.0.0 Type 'exit' to quit")
    while True:
        user = input("You: ")
        if user.lower() == "exit":
            break
        print("KenBot:", chat(user))
