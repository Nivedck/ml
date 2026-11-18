
import random
import torch
import torch.nn as nn
from torch.nn import functional as F

# ============================================================
# 1. HYPERPARAMETERS
# ============================================================

batch_size = 32
block_size = 64
max_iters = 1000
eval_interval = 200
learning_rate = 1e-3
eval_iters = 50

n_embd = 128
n_head = 4
n_layer = 4
dropout = 0.1

temperature = 0.8
top_k = 10

seed = 1337
random.seed(seed)
torch.manual_seed(seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# ============================================================
# 2. FIXED TOY DATASET
#    Varied text, not one sentence repeated 100 times.
# ============================================================

paragraphs = [
    """Hello! How are you doing today?
I am learning how a small transformer works.
A language model predicts the next character or token.
Every training example helps the model discover patterns.""",

    """Python is a popular programming language.
It is widely used in science and machine learning.
Lists store sequences of values.
Functions help programmers organize reusable logic.
Debugging is an important part of software development.""",

    """Machine learning allows computers to learn from data.
A neural network contains layers of trainable parameters.
Training adjusts these parameters to reduce prediction error.
Validation measures performance on held-out examples.
Good data is essential for useful machine learning.""",

    """The sun rises in the east and sets in the west.
Birds build nests in trees and sing in the morning.
Some animals live in forests while others live in oceans.
Plants use sunlight to make their own food.
Rain provides water for rivers, farms, and gardens.""",

    """A transformer uses attention to process a sequence.
Self-attention compares information at different positions.
Causal masking prevents the model from seeing future tokens.
Residual connections help information move through the network.
Layer normalization can make optimization more stable.""",

    """Learning a new skill takes time and patience.
Small steps can lead to meaningful progress.
Mistakes reveal what needs to be improved.
Careful experiments help us test our assumptions.
Curiosity encourages us to keep asking questions.""",

    """The library is quiet during the early morning.
A student opens a book and starts taking notes.
Outside, the rain falls softly on the road.
After lunch, the clouds begin to disappear.
The evening sky turns orange before sunset.""",

    """A computer represents text using numbers.
A tokenizer maps symbols to integer identifiers.
An embedding turns each identifier into a vector.
The model uses these vectors to make predictions.
A decoder converts the predicted identifiers back to text.""",

    """Healthy plants need light, water, and nutrients.
Some seeds grow quickly while others take much longer.
Gardeners observe the soil before watering their plants.
A small garden can attract bees and butterflies.
Fresh vegetables can be harvested when they are ready.""",

    """Science progresses through observation and experimentation.
A hypothesis offers a possible explanation for a question.
An experiment tests the hypothesis under controlled conditions.
Results should be recorded and examined carefully.
New evidence can lead to better explanations.""",

    """The train arrived at the station just before noon.
Travelers collected their bags and checked the signs.
A family waited near the platform entrance.
The conductor announced the next departure.
Soon the train began moving toward the next city.""",

    """A good program should be readable and reliable.
Meaningful variable names make code easier to understand.
Tests help catch mistakes before software is released.
Clear documentation explains how a program should work.
Simple solutions are often easier to maintain.""",

    """The ocean contains many kinds of living organisms.
Coral reefs provide shelter for numerous fish.
Whales travel long distances through open water.
Scientists study marine life to understand ecosystems.
Reducing pollution helps protect aquatic habitats.""",

    """The baker mixed flour, water, and yeast in a bowl.
The dough rested while the oven became warm.
Soon the kitchen smelled like freshly baked bread.
The loaves cooled on a wooden rack.
Everyone gathered around the table for breakfast.""",

    """Data can be stored in files or databases.
A dataset should be checked for errors and duplicates.
Training data teaches a model to make predictions.
Validation data helps compare different model settings.
Test data provides a final independent evaluation.""",

    """Attention assigns different weights to earlier positions.
The weights determine how information is combined.
Multiple attention heads can learn different relationships.
A feed-forward network processes each position independently.
Together these components form a transformer block.""",

    """On Saturday, we walked along a narrow forest path.
Tall trees provided shade from the afternoon sun.
We heard birds calling from the branches above us.
A small stream crossed the trail near a bridge.
We returned home before the sky became dark.""",

    """Probability describes uncertainty about possible outcomes.
A softmax function converts logits into probabilities.
Sampling chooses an outcome from the distribution.
A lower temperature makes predictions more concentrated.
A higher temperature generally creates more varied samples.""",

    """The weather changes from season to season.
Some days are warm, while others are cold and windy.
Clouds can bring rain to dry fields.
A strong breeze moves the leaves along the pavement.
People check the forecast before planning a trip.""",

    """An optimizer updates the parameters of a neural network.
The learning rate controls the size of each update.
Gradients indicate how the loss changes with parameters.
A batch contains examples processed in one training step.
Repeated updates can gradually improve predictions.""",

    """Reading helps us discover ideas from other people.
Writing gives structure to thoughts and observations.
Questions can reveal gaps in our understanding.
Discussion allows different viewpoints to be compared.
Practice turns unfamiliar tasks into familiar routines.""",

    """A robot can use sensors to observe its surroundings.
Its controller processes measurements and selects actions.
Motors translate commands into physical movement.
Testing helps engineers find problems in a design.
Reliable systems must handle unexpected situations.""",

    """The market was busy on a sunny afternoon.
Vendors arranged fruit in colorful wooden boxes.
Customers compared prices and asked about fresh produce.
A musician played a tune beside the entrance.
By evening, the street became quiet again.""",

    """A language model does not automatically understand every topic.
Its behavior depends on its training data and architecture.
A small model has limited capacity and context.
Generated text can be fluent but factually incorrect.
Evaluation is necessary to understand its limitations.""",

    """Graphs represent relationships between connected objects.
A node represents an entity in the graph.
An edge describes a connection between two nodes.
Graph algorithms can find paths and identify communities.
These structures are useful in many applications.""",

    """The classroom windows were open to let in fresh air.
The teacher wrote a question on the board.
Students discussed possible answers in small groups.
They compared their reasoning before sharing their conclusions.
At the end, everyone reviewed the main ideas.""",

    """Images contain patterns such as edges, shapes, and textures.
A convolutional network learns useful visual features.
Audio models can learn patterns in sound waves.
Text models learn statistical patterns in sequences.
Different data types often require different preprocessing.""",

    """A bicycle needs regular maintenance to stay safe.
Check the tires before starting a long ride.
Brakes should work smoothly and reliably.
A helmet provides important protection for the rider.
Following traffic rules helps keep everyone safer.""",

    """Open source projects let people study and improve software.
Contributors report bugs and suggest new features.
Reviewers check proposed changes before they are merged.
Good collaboration depends on clear communication.
Shared tools help developers solve problems together.""",

    """A model checkpoint stores learned parameters.
Saving checkpoints allows training to resume later.
Random seeds can improve experimental reproducibility.
Logs help track losses and configuration changes.
Careful comparisons make experiments more informative.""",
]

# Split by complete paragraphs so validation examples are held out.
random.shuffle(paragraphs)
split_index = max(1, int(0.8 * len(paragraphs)))

train_text = "\n\n".join(paragraphs[:split_index])
val_text = "\n\n".join(paragraphs[split_index:])

# Character-level tokenizer.
# The vocabulary is built from the full toy corpus so every
# validation character has an ID.
text = train_text + "\n\n" + val_text
chars = sorted(set(text))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

def encode(s):
    return [stoi[ch] for ch in s]

def decode(ids):
    return "".join(itos[int(i)] for i in ids)

train_data = torch.tensor(encode(train_text), dtype=torch.long)
val_data = torch.tensor(encode(val_text), dtype=torch.long)

if len(train_data) <= block_size or len(val_data) <= block_size:
    raise ValueError(
        "Training and validation text must each exceed block_size. "
        "Add more paragraphs or reduce block_size."
    )

print(f"Vocabulary size: {vocab_size}")
print(f"Training characters: {len(train_data)}")
print(f"Validation characters: {len(val_data)}")


# ============================================================
# 3. BATCH SAMPLING AND LOSS EVALUATION
# ============================================================

def get_batch(split):
    source = train_data if split == "train" else val_data

    ix = torch.randint(
        len(source) - block_size,
        (batch_size,)
    )

    x = torch.stack([
        source[i:i + block_size] for i in ix
    ])
    y = torch.stack([
        source[i + 1:i + block_size + 1] for i in ix
    ])

    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model):
    results = {}
    model.eval()

    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)

        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()

        results[split] = losses.mean().item()

    model.train()
    return results


# ============================================================
# 4. TRANSFORMER COMPONENTS
# ============================================================

class Head(nn.Module):
    """One causal self-attention head."""

    def __init__(self, head_size):
        super().__init__()

        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size))
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        # Scale by the dimension of each attention head.
        head_size = k.shape[-1]
        wei = (q @ k.transpose(-2, -1)) * (head_size ** -0.5)

        # Prevent positions from looking into the future.
        wei = wei.masked_fill(
            self.tril[:T, :T] == 0,
            float("-inf")
        )

        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        return wei @ v


class MultiHeadAttention(nn.Module):
    """Several attention heads working in parallel."""

    def __init__(self, num_heads, head_size):
        super().__init__()

        self.heads = nn.ModuleList([
            Head(head_size) for _ in range(num_heads)
        ])

        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, embd):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(embd, 4 * embd),
            nn.GELU(),
            nn.Linear(4 * embd, embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block with pre-layer normalization."""

    def __init__(self, embd, num_heads):
        super().__init__()

        head_size = embd // num_heads

        self.sa = MultiHeadAttention(num_heads, head_size)
        self.ffwd = FeedForward(embd)

        self.ln1 = nn.LayerNorm(embd)
        self.ln2 = nn.LayerNorm(embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


# ============================================================
# 5. GPT LANGUAGE MODEL
# ============================================================

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.token_embedding_table = nn.Embedding(
            vocab_size, n_embd
        )
        self.position_embedding_table = nn.Embedding(
            block_size, n_embd
        )

        self.blocks = nn.Sequential(*[
            Block(n_embd, n_head) for _ in range(n_layer)
        ])

        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        if T > block_size:
            raise ValueError("Input sequence exceeds block_size.")

        tok_emb = self.token_embedding_table(idx)
        pos = torch.arange(T, device=idx.device)
        pos_emb = self.position_embedding_table(pos)

        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None

        if targets is not None:
            B, T, C = logits.shape

            loss = F.cross_entropy(
                logits.reshape(B * T, C),
                targets.reshape(B * T)
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens,
        temperature=0.8,
        top_k=10
    ):
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero.")

        self.eval()

        for _ in range(max_new_tokens):
            # Keep only the most recent context window.
            idx_cond = idx[:, -block_size:]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Optional top-k sampling.
            if top_k is not None:
                k = min(top_k, logits.size(-1))
                threshold = torch.topk(logits, k).values[:, -1:]
                logits = logits.masked_fill(
                    logits < threshold,
                    float("-inf")
                )

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)

        return idx


# ============================================================
# 6. TRAINING
# ============================================================

model = GPTLanguageModel().to(device)

num_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {num_params:,}")

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=0.01
)

print("\nTraining model from scratch...")

for step in range(max_iters):
    if step % eval_interval == 0:
        losses = estimate_loss(model)

        print(
            f"step {step:4d}: "
            f"train loss {losses['train']:.4f}, "
            f"val loss {losses['val']:.4f}"
        )

    xb, yb = get_batch("train")

    _, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    # Keep unusually large gradients from destabilizing training.
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    optimizer.step()

# Final loss measurement after the last update.
losses = estimate_loss(model)
print(
    f"final:     train loss {losses['train']:.4f}, "
    f"val loss {losses['val']:.4f}"
)


# ============================================================
# 7. TEXT GENERATION
# ============================================================

model.eval()

prompts = [
    "A language model",
    "Machine learning",
    "The transformer",
    "India",
    "what is "
]

print("\nGenerated Text from Scratch Model:")

for prompt in prompts:
    context = torch.tensor(
        [encode(prompt)],
        dtype=torch.long,
        device=device
    )

    output = model.generate(
        context,
        max_new_tokens=150,
        temperature=temperature,
        top_k=top_k
    )

    print(f"\nPrompt: {prompt}")
    print(decode(output[0].tolist()))
    print("-" * 60)
