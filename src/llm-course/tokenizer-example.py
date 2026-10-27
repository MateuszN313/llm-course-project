from transformers import AutoTokenizer

MODEL_NAME_V1 = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer_v1 = AutoTokenizer.from_pretrained(MODEL_NAME_V1)
MODEL_NAME_V2 = "bert-base-cased"
tokenizer_v2 = AutoTokenizer.from_pretrained(MODEL_NAME_V2)

samples = [
    "kot",
    "nieprzewidywalność",
    "Large language model",
    "Duży model językowy",
    "print('Hello!')",
    "LLM-y kosztują 0.01 USD",
]
#🤖

print("{")

for name, tokenizer in [(MODEL_NAME_V1, tokenizer_v1), (MODEL_NAME_V2, tokenizer_v2)]:
    
    print(f" \"{name}\": [")
    for text in samples:
        ids = tokenizer.encode(text, add_special_tokens=False)
        pieces = [tokenizer.decode([token_id]) for token_id in ids]
        pieces = list(map(lambda x: f"\"{x}\"", pieces))
        print("  {")
        print(f"   \"TEXT\":   \"{text}\",")
        print(f"   \"IDS\":    {ids},")
        print(f"   \"PIECES\": [{' ,'.join(pieces)}],")
        print(f"   \"COUNT\":  {len(ids)}")
        print("  },")
    print(" ],")

print("}")