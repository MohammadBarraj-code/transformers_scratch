VOCAB_SIZE = 50260
D_MODEL = 64
NUM_HEADS = 4
NUM_LAYERS = 2
D_FF = 128
EPOCHS = 2000


source_sentences = ["hello", "the cat", "goodbye", "cat", "dog", "blue", "red",
                    "green", "house", "the cat is blue"]

target_sentences = ["bonjour", "le chat", "au revoir", "chat", "chien", "bleu",
                    "rouge", "vert", "maison", "le chat est bleu"]


import matplotlib.pyplot as plt
import transformer
import torch, torch.nn as nn
import tokenizer, attention, feed_forward_nn

tokenizer = tokenizer.Tokenizer()

def generate_batch(source_sentences, target_sentences, tokenizer):
  src = tokenizer.batch_encoder(source_sentences)["input_ids"]
  tgt = tokenizer.batch_encoder(target_sentences)["input_ids"]

  return src, tgt


model = transformer.Transformer(VOCAB_SIZE, VOCAB_SIZE, D_MODEL, NUM_HEADS, NUM_LAYERS, D_FF)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.tokenizer.pad_token_id)

for epoch in range(EPOCHS):
  src, target = generate_batch(source_sentences, target_sentences, tokenizer)
  tgt_input = target[:, :-1]
  tgt_output = target[:, 1:]

  src_mask = (src != 50259).unsqueeze(1).unsqueeze(2)
  tgt_padding_mask = (tgt_input != tokenizer.tokenizer.pad_token_id).unsqueeze(1).unsqueeze(2)

  tgt_padding_mask = (tgt_input != 50259).unsqueeze(1).unsqueeze(2)


  tgt_length = tgt_input.size(1)

  tgt_causal_mask = torch.tril(
    torch.ones(tgt_length, tgt_length, dtype=torch.bool)
  )

  tgt_mask = tgt_padding_mask & tgt_causal_mask

  output, _ = model(src, tgt_input, src_mask, tgt_mask)

  loss = criterion(output.view(-1, VOCAB_SIZE), tgt_output.reshape(-1))

  optimizer.zero_grad()
  loss.backward()
  optimizer.step()


  if epoch % 10 == 0:
    print(f"Epoch {epoch}. Loss: {loss.item()}")

model.eval()

def translate(sentence):

    src_ids = tokenizer.encode(sentence)

    src = torch.tensor(
        [src_ids],
        dtype=torch.long
    )

    src_mask = (
        src != tokenizer.tokenizer.pad_token_id
    ).unsqueeze(1).unsqueeze(2)

    output_ids = [
        tokenizer.tokenizer.bos_token_id
    ]

    generation_steps = []

    with torch.no_grad():

        for _ in range(10):

            tgt_input = torch.tensor(
                [output_ids],
                dtype=torch.long
            )

            tgt_length = tgt_input.size(1)

            tgt_causal_mask = torch.tril(
                torch.ones(
                    tgt_length,
                    tgt_length,
                    dtype=torch.bool
                )
            )

            output, attention = model(
                src,
                tgt_input,
                src_mask,
                tgt_causal_mask
            )

            # Predict next token
            next_token = output[:, -1, :].argmax(
                dim=-1
            ).item()

            output_ids.append(next_token)

            generation_steps.append(
                tokenizer.decode(output_ids)
            )

            if next_token == tokenizer.tokenizer.eos_token_id:
                break

    return (
        tokenizer.decode(output_ids),
        generation_steps,
        attention,
        src_ids,
        output_ids
    )


# ============================================================
# TRANSLATION
# ============================================================

translation, steps, attention, src_ids, output_ids = translate(
    "the cat is blue"
)

print("INPUT:", "the cat is blue")
print("OUTPUT:", translation)

print("\nGENERATION:")

for i, step in enumerate(steps):
    print(f"Step {i + 1}: {step}")


# ============================================================
# ATTENTION HEATMAP
# ============================================================

# Average all attention heads
# Shape: [target_length, source_length]

attention_map = attention[0].mean(dim=0).cpu().numpy()


# ============================================================
# TOKEN LABELS
# ============================================================

source_tokens = [
    tokenizer.tokenizer.decode([token])
    for token in src_ids
]

# Attention was calculated BEFORE the final predicted token
# was appended, so remove the final token from the labels.

target_tokens = [
    tokenizer.tokenizer.decode([token])
    for token in output_ids[:-1]
]


# ============================================================
# HEATMAP
# ============================================================

plt.figure(figsize=(10, 7))

plt.imshow(
    attention_map,
    cmap="viridis",
    aspect="auto"
)

plt.xticks(
    range(len(source_tokens)),
    source_tokens
)

plt.yticks(
    range(len(target_tokens)),
    target_tokens
)

plt.xlabel("Source Tokens")
plt.ylabel("Decoder Input Tokens")

plt.title(
    "Transformer Cross-Attention"
)

plt.colorbar(
    label="Attention Weight"
)

plt.tight_layout()

plt.show()