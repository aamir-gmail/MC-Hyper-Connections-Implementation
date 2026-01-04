# A tiny implemetation of paper
# Manifold-Constrained Hyper-Connections by Deep Seek
# https://www.alphaxiv.org/abs/2512.24880/sso-callback
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import requests
import os

# Character level transformer
# --- Configuration ---
BATCH_SIZE = 32
BLOCK_SIZE = 128  # Context length
LEARNING_RATE = 3e-4
EVAL_INTERVAL = 100
# mHC Specifics
N_STREAMS = 4  # Expansion rate n [cite: 373]
EMBED_DIM = 128  # Dimension C , reduce this to 64 if you a GPU contr
N_LAYERS = 4  # Number of Attention+MLP pairs , Reduce this to 4
SINKHORN_ITERS = 20  # [cite: 276]
MAX_ITERS = 12000       # Extended training steps
WARMUP_STEPS = 100     # Warmup period
GRAD_CLIP = 0.8        # Critical for preventing explosions
WEIGHT_DECAY = 0.1     # Prevents unbounded weight growth
DROPOUT = 0.05

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'



torch.manual_seed(1337) # For Reproducibility


# --- Part 1: The Manifold Constraint (Sinkhorn) ---

class SinkhornProjection(nn.Module):
    """
    Projects raw logits onto the Birkhoff polytope (Doubly Stochastic).
    Ref: Eq. (9) and (19) [cite: 272, 318]
    """

    def __init__(self, iterations=SINKHORN_ITERS):
        super().__init__()
        self.iterations = iterations

    def forward(self, x):
        # Numerical stability: subtract max before exp
        x_safe = x - x.max(dim=-1, keepdim=True).values
        matrix = torch.exp(x_safe)

        for _ in range(self.iterations):
            # Row normalization
            matrix = matrix / (matrix.sum(dim=-1, keepdim=True) + 1e-6)
            # Column normalization
            matrix = matrix / (matrix.sum(dim=-2, keepdim=True) + 1e-6)
        return matrix


# --- Part 2: The mHC Wrapper Layer ---

class MHCWrapper(nn.Module):
    """
    Wraps an arbitrary layer F (Attn or MLP) with Manifold-Constrained Hyper-Connections.
    Handles reshaping so Attention gets (B, T, C) while coefficients are generated per token.
    """

    def __init__(self, layer_f, dim, n_streams=N_STREAMS):
        super().__init__()
        self.layer_f = layer_f
        self.dim = dim
        self.n = n_streams
        self.total_dim = dim * n_streams

        # Coefficient Generators
        self.coef_norm = nn.RMSNorm(self.total_dim)
        self.proj_pre = nn.Linear(self.total_dim, n_streams)
        self.proj_post = nn.Linear(self.total_dim, n_streams)
        self.proj_res = nn.Linear(self.total_dim, n_streams * n_streams)

        # Gating Factors
        self.alpha_pre = nn.Parameter(torch.tensor(0.01))
        self.alpha_post = nn.Parameter(torch.tensor(0.01))
        self.alpha_res = nn.Parameter(torch.tensor(0.01))

        self.sinkhorn = SinkhornProjection()
        self.layer_norm = nn.RMSNorm(dim)

    def get_dynamic_mappings(self, x_flat):
        # x_flat is (Batch*Time, n*C)
        x_norm = self.coef_norm(x_flat)

        # H_pre: (B*T, 1, n)
        h_pre_logits = self.alpha_pre * self.proj_pre(x_norm)
        H_pre = torch.sigmoid(h_pre_logits).unsqueeze(1)

        # H_post: (B*T, 1, n)
        h_post_logits = self.alpha_post * self.proj_post(x_norm)
        H_post = 2 * torch.sigmoid(h_post_logits).unsqueeze(1)

        # H_res: (B*T, n, n)
        h_res_logits = self.alpha_res * self.proj_res(x_norm)
        h_res_logits = h_res_logits.view(-1, self.n, self.n)
        H_res = self.sinkhorn(h_res_logits)

        return H_pre, H_post, H_res

    def forward(self, x):
        # FIX: Accept 4D input (Batch, Time, Streams, Dim)
        B, T, n, C = x.shape

        # 1. Flatten for Coefficient Generation (Per Token)
        # Reshape to (Batch*Time, n*C)
        x_flat = x.view(B * T, -1)
        H_pre, H_post, H_res = self.get_dynamic_mappings(x_flat)

        # 2. Compute Layer Branch
        # Flatten x to (Batch*Time, n, C) for matrix multiplication
        x_per_token = x.view(B * T, n, C)

        # Pre-Mapping: (B*T, 1, n) @ (B*T, n, C) -> (B*T, 1, C)
        x_aggregated = torch.einsum('bjn, bnc -> bjc', H_pre, x_per_token).squeeze(1)

        # FIX: Reshape to (Batch, Time, C) for the Sub-Layer (Attention needs T)
        x_for_layer = x_aggregated.view(B, T, C)

        # Apply Sub-Layer (Attn/MLP)
        x_layer_out = self.layer_f(self.layer_norm(x_for_layer))  # Returns (B, T, C)

        # Flatten back for Post-Mapping: (Batch*Time, C)
        x_layer_out_flat = x_layer_out.view(B * T, C)

        # Post-Mapping: Broadcast back to streams
        # (B*T, n, 1) @ (B*T, 1, C) -> (B*T, n, C)
        x_branch = torch.einsum('bjn, bc -> bnc', H_post.transpose(1, 2), x_layer_out_flat)

        # 3. Compute Residual Highway
        # (B*T, n, n) @ (B*T, n, C) -> (B*T, n, C)
        x_highway = torch.einsum('bnm, bmc -> bnc', H_res, x_per_token)

        # Sum and Restore 4D Shape
        x_next = x_highway + x_branch
        return x_next.view(B, T, n, C)


# --- Part 3: Standard Transformer Components ---

class CausalSelfAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.num_heads = 4
        self.head_dim = dim // 4
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        # NEW: Attention Dropout and Resid Dropout
        self.attn_dropout = nn.Dropout(DROPOUT)
        self.resid_dropout = nn.Dropout(DROPOUT)

        self.register_buffer("bias", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE))
                             .view(1, 1, BLOCK_SIZE, BLOCK_SIZE))

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        # Apply Dropout to attention weights
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Apply Dropout to output projection
        return self.resid_dropout(self.proj(y))

class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.SiLU(),
            nn.Linear(4 * dim, dim),
            nn.Dropout(DROPOUT) # NEW: Prevent co-adaptation of neurons
        )

    def forward(self, x):
        return self.net(x)

# --- Part 4: The Full Language Model ---

class MHCLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, EMBED_DIM * N_STREAMS)
        self.position_embedding = nn.Embedding(BLOCK_SIZE, EMBED_DIM * N_STREAMS)

        self.layers = nn.ModuleList([])
        for _ in range(N_LAYERS):
            # Sub-layer 1: Attention wrapped in mHC
            self.layers.append(MHCWrapper(
                CausalSelfAttention(EMBED_DIM), dim=EMBED_DIM
            ))

            # Sub-layer 2: MLP wrapped in mHC
            self.layers.append(MHCWrapper(
                MLP(EMBED_DIM), dim=EMBED_DIM
            ))

        self.final_norm = nn.RMSNorm(EMBED_DIM * N_STREAMS)
        self.head = nn.Linear(EMBED_DIM * N_STREAMS, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # Embeddings
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=DEVICE))
        x = tok_emb + pos_emb
        x = x.view(B, T, N_STREAMS, EMBED_DIM)
        # Pass through mHC Layers
        for layer in self.layers:
            x = layer(x)

        # Flatten back to (Batch, Time, n*C) for output
        x = x.view(B, T, -1)
        x = self.final_norm(x)
        logits = self.head(x)

        if targets is None:
            loss = None
        else:
            B, T, V = logits.shape
            logits = logits.view(B * T, V)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK_SIZE:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# --- Helper: Data Loader ---
def get_data(filename="timemachine.txt"):
    # For better results increase the size of the text by 10x
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        # this is a small file for testing only.
        url = "https://www.gutenberg.org/cache/epub/35/pg35.txt"  # The Time Machine
        res = requests.get(url)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(res.text)

    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()
    return text


# --- Main Execution ---
if __name__ == "__main__":
    # 1. Prepare Data
    text = get_data()
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]


    def get_batch(split):
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
        x = torch.stack([data[i:i + BLOCK_SIZE] for i in ix])
        y = torch.stack([data[i + 1:i + BLOCK_SIZE + 1] for i in ix])
        x, y = x.to(DEVICE), y.to(DEVICE)
        return x, y


    # 2. Init Model
    print(f"Initializing mHC Model with {N_STREAMS} streams...")
    model = MHCLanguageModel(vocab_size)
    model = model.to(DEVICE)

    # 1. OPTIMIZER WITH WEIGHT DECAY
    # We separate params: Apply weight decay to matmul weights, but NOT to biases/layernorms/embeddings
    param_dict = {pn: p for pn, p in model.named_parameters()}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': WEIGHT_DECAY},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=LEARNING_RATE, betas=(0.9, 0.95))


    # 2. LEARNING RATE SCHEDULER (Cosine Decay with Warmup)
    def get_lr(it):
        # 1) Linear Warmup
        if it < WARMUP_STEPS:
            return LEARNING_RATE * (it + 1) / (WARMUP_STEPS + 1)
        # 2) If it > MAX_ITERS, return min lr
        if it > MAX_ITERS:
            return LEARNING_RATE * 0.1
        # 3) Cosine Decay
        decay_ratio = (it - WARMUP_STEPS) / (MAX_ITERS - WARMUP_STEPS)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return LEARNING_RATE * 0.1 + coeff * (LEARNING_RATE * 0.9)


    # --- ROBUST TRAINING LOOP ---
    print("Starting training with Stability fixes...")

    for iter in range(MAX_ITERS):

        # A. Apply LR Schedule
        lr = get_lr(iter)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # B. Evaluation
        if iter % EVAL_INTERVAL == 0:
            model.eval()
            with torch.no_grad():
                losses = torch.zeros(50)
                for k in range(50):
                    X, Y = get_batch('val')
                    logits, loss = model(X, Y)
                    losses[k] = loss.item()
            print(f"Step {iter}: Val Loss {losses.mean():.4f} | LR: {lr:.2e}")
            model.train()

        # C. Forward & Backward
        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # D. GRADIENT CLIPPING (The Anti-Explosion Shield)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        optimizer.step()

    # 4. Generate Text
    print("\n--- Generating Text ---")
    context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    print(decode(model.generate(context, max_new_tokens=300)[0].tolist()))
