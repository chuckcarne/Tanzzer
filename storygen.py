import re
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_confession_story(max_words=50):  # Set max_words to 50
    model_name = "gpt2-medium"
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    prompt = (
        "I'm a student at Davis High School, and I have a secret to confess. "
        "I'm posting this anonymously for everyone at school to see. Here it goes:\n\n"
    )

    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    max_length = 400
    output = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        top_p=0.85,
        temperature=0.7,
        top_k=40,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        repetition_penalty=1.2,
    )

    story = tokenizer.decode(output[0], skip_special_tokens=True)
    confession = story[len(prompt):].strip()

    # Cap the word count
    words = confession.split()
    if len(words) > max_words:
        confession = ' '.join(words[:max_words])  # Limit to max_words

    # Fix cut-off issue: trim to the nearest sentence boundary
    sentence_endings = re.compile(r'([.!?])')
    last_punctuation_pos = max([confession.rfind(p) for p in ['.', '!', '?']])

    if last_punctuation_pos != -1:
        confession = confession[:last_punctuation_pos + 1]

    return confession

if __name__ == "__main__":
    for _ in range(3):
        print(generate_confession_story(max_words=50))  # Limit to 50 words
        print("\n\n")
    
