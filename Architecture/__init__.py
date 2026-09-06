VOCAB_SIZE = 20
D_MODEL = 64
NUM_HEADS = 4
NUM_LAYERS = 2
D_FF = 128
SEQ_LEN = 10
BATCH_SIZE = 32
#prev 1000 epochs but trying to increase epoch count to see if we can overfit toy model 100%
#output on 1000 epochs reisudal 2 src_mask none: [0,1,2,3,4,5,6,7,8,10,9]
#output on 2000 epochs:[0,1,2,3,4,5,6,7,8,9,10]
EPOCHS = 2000

import torch, torch.nn as nn
import transformer

def generate_batch(batch_size, seq_len, vocab_size):
  src = torch.randint(1, vocab_size, (batch_size, seq_len))
  tgt_input = torch.cat([torch.zeros(batch_size, 1, dtype=torch.long), src], dim=1)   # ← ADD [0, src...]
  tgt_output = torch.cat([src, torch.zeros(batch_size, 1, dtype=torch.long)], dim=1)  # ← ADD [src..., 0]
  return src, tgt_input, tgt_output

model = transformer.Transformer(VOCAB_SIZE, VOCAB_SIZE, D_MODEL, NUM_HEADS, NUM_LAYERS, D_FF)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
  # For each epoch:
  #   1. Generate a batch
  src, tgt_input, tgt_output = generate_batch(BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
  #   2. Forward pass (feed src and tgt[:-1] as decoder input)
  '''
  Look at everything except the last token
  '''
  tgt_seq_len = tgt_input.shape[1]
  tgt_mask = torch.tril(
    torch.ones(tgt_seq_len, tgt_seq_len, dtype=torch.bool)
    )
  output = model(src, tgt_input, target_mask=tgt_mask)
  #   3. Compare the output to tgt[1:] (Shifted by one - next token prediction)
  '''
  Spanish target: [<start>, Hola, mundo, <end>]

  Position 0:
  > Decoder sees <start> (from tgt[:-1])
  > Should predict: Hola <- This is tgt[1] the first element of tgt[1:]

  Position 1:
  > Decoder sees <start> <Hola> (from tgt[:-1])
  > Should predict: mundo <- This is tgt[2] the second element of tgt[1:]
  '''
  loss = criterion(output.view(-1, VOCAB_SIZE), tgt_output.reshape(-1))

  #   4: Backprop and update
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()

  if epoch % 10 == 0:
    print(f"Epoch {epoch}, Loss: {loss.item()}")


    # Post Training

model.eval()

test_src = torch.tensor([[1,2,3,4,5,6,7,8,9,10]])

output_seq = [0]

with torch.no_grad():

    for i in range(SEQ_LEN):

        tgt_input = torch.tensor([output_seq])

        tgt_seq_len = tgt_input.size(1)

        tgt_mask = torch.tril(
            torch.ones(
                tgt_seq_len,
                tgt_seq_len,
                dtype=torch.bool
            )
        )

        output = model(
            test_src,
            tgt_input,
            target_mask=tgt_mask
        )

        next_token = output[:, -1, :].argmax(dim=-1).item()

        output_seq.append(next_token)

print(output_seq)