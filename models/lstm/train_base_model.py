import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle
import os
from tqdm import tqdm

from dataset import WikiDataset
from model import LSTMLanguageModel


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEQ_LEN = 50
BATCH_SIZE = 128
EPOCHS = 2
LR = 0.001

DATA_FILE = "data/processed/wiki_sentences.txt"
VOCAB_FILE = "data/processed/vocab.pkl"
SAVE_PATH = "models/lstm/base_model.pt"
CHECKPOINT_PATH = "models/lstm/checkpoint.pt"


def train():

    dataset = WikiDataset(DATA_FILE, VOCAB_FILE, SEQ_LEN, max_tokens=2_000_000)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    with open(VOCAB_FILE, "rb") as f:
        vocab = pickle.load(f)

    vocab_size = len(vocab["word2idx"])

    model = LSTMLanguageModel(vocab_size).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    start_epoch = 0
    if os.path.exists(CHECKPOINT_PATH):
        print(f"Resuming from checkpoint: {CHECKPOINT_PATH}")
        ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"]
        print(f"Resuming from epoch {start_epoch + 1}")

    if start_epoch >= EPOCHS:
        print("Training already complete.")
        if not os.path.exists(SAVE_PATH):
            torch.save(model.state_dict(), SAVE_PATH)
            print("Model saved to:", SAVE_PATH)
        return

    for epoch in range(start_epoch, EPOCHS):
        total_loss = 0

        for x, y in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()
            logits, _ = model(x)

            loss = criterion(
                logits.view(-1, vocab_size),
                y.view(-1)
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")

        os.makedirs("models/lstm", exist_ok=True)
        torch.save({
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        }, CHECKPOINT_PATH)
        print(f"Checkpoint saved after epoch {epoch+1}")

    torch.save(model.state_dict(), SAVE_PATH)
    print("Model saved to:", SAVE_PATH)
    os.remove(CHECKPOINT_PATH)
    print("Checkpoint removed.")


if __name__ == "__main__":
    # ── Inspection mode — does NOT retrain ───────────────────────
    # If base_model.pt exists: load it and print stats.
    # If missing: print training instructions.
    #
    # To actually train (takes ~1-2 hours on CPU, ~20 min on GPU):
    #   cd models/lstm
    #   python train_base_model.py   ← remove this guard first
    # ─────────────────────────────────────────────────────────────
    import sys
    sys.path.insert(0, ".")

    if not os.path.exists(SAVE_PATH):
        print(f"No trained model found at {SAVE_PATH}")
        if os.path.exists(CHECKPOINT_PATH):
            print(f"Partial checkpoint found at {CHECKPOINT_PATH} — training was interrupted.")
            ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
            print(f"  Completed epochs : {ckpt['epoch']} / {EPOCHS}")
        else:
            print("No checkpoint found either. Run train() to start training.")
        sys.exit(0)

    with open(VOCAB_FILE, "rb") as f:
        vocab = pickle.load(f)
    vocab_size = len(vocab["word2idx"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = LSTMLanguageModel(vocab_size).to(device)
    model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model             : {SAVE_PATH}")
    print(f"Vocab size        : {vocab_size}")
    print(f"Device            : {device}")
    print(f"Total parameters  : {total_params:,}")
    print(f"Training config   : epochs={EPOCHS}, lr={LR}, batch={BATCH_SIZE}, seq_len={SEQ_LEN}")
