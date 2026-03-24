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
    train()
