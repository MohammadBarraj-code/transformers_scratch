from transformers import GPT2Tokenizer
import torch
class Tokenizer:
    def __init__(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        self.tokenizer.add_special_tokens({
            "bos_token": "<SOS>",
            "eos_token": "<EOS>",
            "pad_token": "<PAD>"
        })

    def encode(self, text):
        token_ids = self.tokenizer.encode(text)

        return [
            self.tokenizer.bos_token_id,
            *token_ids,
            self.tokenizer.eos_token_id
        ]

    def batch_encoder(self, texts):
        encoded = [self.encode(text) for text in texts]

        return self.tokenizer.pad(
            {"input_ids":encoded},
            padding=True,
            return_tensors="pt"
        )

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)


"""tokenizer = Tokenizer()

texts = ["Hello World", "Transformers are cool"]

batch = tokenizer.batch_encoder(texts)

print(batch["input_ids"])
print(batch["attention_mask"])"""