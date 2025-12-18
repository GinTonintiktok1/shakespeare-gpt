# ========================================
# shake_py.py - CODICE COMPLETO
# ========================================

# FIX OMP
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = 'cpu'

print("🎭 Shakespeare GPT - Generator")
print("="*60)

# ========================================
# POSITIONAL ENCODING
# ========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# ========================================
# MULTI-HEAD ATTENTION
# ========================================
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        Q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.W_o(out)
        
        return out

# ========================================
# TRANSFORMER BLOCK
# ========================================
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask):
        x = x + self.dropout(self.attn(self.ln1(x), mask))
        x = x + self.dropout(self.ff(self.ln2(x)))
        return x

# ========================================
# GPT MODEL
# ========================================
class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, 
                 max_seq_len, dropout=0.1):
        super().__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        batch_size, seq_len = x.shape
        
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).unsqueeze(0).unsqueeze(0)
        
        x = self.token_embedding(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x, mask)
        
        x = self.ln_f(x)
        logits = self.head(x)
        
        return logits

# ========================================
# GENERAZIONE
# ========================================
@torch.no_grad()
def generate(model, prompt, max_new_tokens=300, temperature=0.8, top_k=40):
    model.eval()
    
    tokens = torch.tensor([ord(c) for c in prompt], dtype=torch.long).unsqueeze(0).to(device)
    
    for _ in range(max_new_tokens):
        tokens_cond = tokens if tokens.size(1) <= 128 else tokens[:, -128:]
        
        logits = model(tokens_cond)
        logits = logits[:, -1, :] / temperature
        
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        tokens = torch.cat([tokens, next_token], dim=1)
    
    return ''.join([chr(int(t)) for t in tokens[0].cpu().numpy()])

# ========================================
# MAIN
# ========================================
if __name__ == '__main__':
    print("📦 Loading model...")
    
    # Crea modello
    model = GPT(
        vocab_size=256,
        d_model=128,
        num_heads=8,
        num_layers=6,
        d_ff=512,
        max_seq_len=128,
        dropout=0.1
    ).to(device)
    
    print(f"✅ Model created: {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    
    # Carica checkpoint
    checkpoint_path = 'checkpoint_epoch_5.pth'
    
    try:
        print(f"📂 Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print("✅ Checkpoint loaded successfully!")
        print(f"📊 Epoch: {checkpoint.get('epoch', 'N/A')}")
        
        train_loss = checkpoint.get('train_loss')
        val_loss = checkpoint.get('val_loss')
        
        if train_loss:
            print(f"📉 Train Loss: {train_loss:.4f}")
        if val_loss:
            print(f"📉 Val Loss: {val_loss:.4f}")
        
    except FileNotFoundError:
        print(f"\n❌ ERROR: File '{checkpoint_path}' not found!")
        print("\n💡 Steps to fix:")
        print("   1. Download checkpoint_epoch_5.pth from Google Drive")
        print("   2. Put it in the same folder as this script")
        print(f"   3. Current folder: {os.getcwd()}")
        exit(1)
    
    except Exception as e:
        print(f"\n❌ ERROR loading checkpoint: {e}")
        exit(1)
    
    # Genera samples
    print("\n" + "="*60)
    print("🎬 GENERATING SHAKESPEARE TEXT")
    print("="*60)
    
    test_prompts = [
        ("ROMEO:", 0.8),
        ("JULIET:", 0.9),
        ("HAMLET:", 0.7),
    ]
    
    for prompt, temp in test_prompts:
        print(f"\n📝 Prompt: '{prompt}' (temperature={temp})")
        print("-"*60)
        
        try:
            generated = generate(model, prompt, max_new_tokens=350, temperature=temp, top_k=40)
            print(generated)
        except Exception as e:
            print(f"❌ Generation error: {e}")
        
        print("="*60)
    
    # Salva modello production
    print("\n💾 Saving production model...")
    try:
        torch.save(model.state_dict(), 'shakespeare_gpt_production.pth')
        print("✅ Saved: shakespeare_gpt_production.pth")
    except Exception as e:
        print(f"❌ Save error: {e}")
    
    print("\n🎉 ALL DONE!")
    print("✨ Your Shakespeare GPT is ready to use!")
