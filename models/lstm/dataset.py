import torch
from torch.utils.data import Dataset
import pickle

class WikiDataset(Dataset):
    def __init__(self, file_path, vocab_path, seq_len=50, max_tokens=2_000_000):
        self.seq_len = seq_len

        with open(vocab_path, "rb") as f:
            vocab = pickle.load(f)

        self.word2idx = vocab["word2idx"]

        token_stream = []
        total_tokens = 0

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                tokens = line.strip().split()
                if len(tokens) >= 2:
                    tokens = ["<SOS>"] + tokens + ["<EOS>"]
                    indices = [
                        self.word2idx.get(t, self.word2idx["<UNK>"])
                        for t in tokens
                    ]
                    token_stream.extend(indices)
                    total_tokens += len(indices)

                    if total_tokens >= max_tokens:
                        break

        self.data = torch.tensor(token_stream, dtype=torch.long)

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + 1 : idx + self.seq_len + 1]
        return x, y
