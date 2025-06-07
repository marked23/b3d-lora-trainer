from torch.utils.data import Dataset

class CodeTextDataset(Dataset):
    def __init__(self, file_path, tokenizer, block_size=256):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        # Tokenize all at once for efficiency
        tokens = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"].squeeze()
        # Chop into windows (block_size + 1 for target)
        self.block_size = block_size
        stride = block_size // 3
        self.samples = []
        for i in range(0, len(tokens) - block_size, stride):
            self.samples.append(tokens[i:i + block_size + 1])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = self.samples[idx][:-1]    # Input tokens
        y = self.samples[idx][1:]     # Target tokens (next token for each position)
        return x, y
