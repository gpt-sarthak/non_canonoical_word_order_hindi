"""
model.py

LSTM language model used for both base and adaptive surprisal.

Architecture:
    Embedding(vocab_size, 256)
    → LSTM(256, hidden=256, layers=2, dropout=0.3)
    → Linear(256, vocab_size)

Input  : token ID tensor of shape (batch, seq_len)
Output : logits tensor (batch, seq_len, vocab_size)  +  hidden state

Trained on Hindi Wikipedia (data/processed/wiki_sentences.txt).
Saved to models/lstm/base_model.pt after training.
"""

import torch
import torch.nn as nn


class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_size=256, hidden_size=256, num_layers=2, dropout=0.3):
        super().__init__()

        # Embedding layer: maps token IDs → dense vectors
        self.embedding = nn.Embedding(vocab_size, embed_size)

        # 2-layer LSTM: processes the embedded sequence left-to-right
        # dropout=0.3 applied between layers (not on final layer output)
        self.lstm = nn.LSTM(
            embed_size,
            hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )

        # Linear projection: maps hidden state → vocabulary logits
        # Used with cross-entropy loss during training and
        # log-softmax during surprisal scoring
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        logits = self.fc(out)
        return logits, hidden


# ─────────────────────────────────────────────────────────────
# __main__ — inspect saved model (no training)
#
# Loads base_model.pt and prints architecture + parameter count.
# Does NOT retrain. If the file is missing, prints instructions.
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, ".")

    import torch
    import pickle

    VOCAB_PATH = "data/processed/vocab.pkl"
    MODEL_PATH = "models/lstm/base_model.pt"

    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}")
        print("To train: cd models/lstm && python train_base_model.py")
        sys.exit(1)

    with open(VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)
    vocab_size = len(vocab["word2idx"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = LSTMLanguageModel(vocab_size)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded from : {MODEL_PATH}")
    print(f"Vocab size        : {vocab_size}")
    print(f"Device            : {device}")
    print(f"Total parameters  : {total_params:,}")
    print(f"\nArchitecture:\n{model}")
