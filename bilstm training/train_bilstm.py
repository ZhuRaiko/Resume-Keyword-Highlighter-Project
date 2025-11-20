import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ----------------------------------------------------
# LOAD DATASET (NO MANUAL CLEANING NEEDED)
# ----------------------------------------------------
df = pd.read_csv("self_promotion_dataset.csv")

sentences = df["sentence"].astype(str).tolist()
labels = df["label"].astype(int).tolist()

# ----------------------------------------------------
# LOAD BERT TOKENIZER (SAME AS YOUR STREAMLIT APP)
# ----------------------------------------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

encodings = tokenizer(
    sentences,
    truncation=True,
    padding=True,
    max_length=128,
    return_tensors="pt"
)

input_ids = encodings["input_ids"]
attention_masks = encodings["attention_mask"]
labels_tensor = torch.tensor(labels)

dataset = TensorDataset(input_ids, attention_masks, labels_tensor)

# ----------------------------------------------------
# TRAIN/VALIDATION SPLIT
# ----------------------------------------------------
train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)

train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
val_loader = DataLoader(val_data, batch_size=8)

# ----------------------------------------------------
# BI-LSTM MODEL
# ----------------------------------------------------
class BiLSTM(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(30522, 128)  # matches bert-base tokenizer vocab
        self.lstm = nn.LSTM(128, hidden_dim, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        x, _ = self.lstm(x)
        x = self.dropout(x[:, -1, :])
        return torch.sigmoid(self.fc(x))

device = "cuda" if torch.cuda.is_available() else "cpu"
model = BiLSTM().to(device)

# ----------------------------------------------------
# TRAINING SETUP
# ----------------------------------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
loss_fn = nn.BCELoss()

# ----------------------------------------------------
# TRAINING LOOP
# ----------------------------------------------------
EPOCHS = 4
for epoch in range(EPOCHS):
    model.train()
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for batch in loop:
        input_ids, _, y = [b.to(device) for b in batch]
        preds = model(input_ids).flatten()
        loss = loss_fn(preds, y.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())

    # validation accuracy
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids, _, y = [b.to(device) for b in batch]
            preds = (model(input_ids).flatten() > 0.5).long()
            correct += (preds == y).sum().item()
            total += len(y)

    print(f"Validation Accuracy: {correct / total :.4f}")

# ----------------------------------------------------
# SAVE TRAINED WEIGHTS
# ----------------------------------------------------
torch.save(model.state_dict(), "bilstm_selfpromo.pt")
print("\nTraining complete → bilstm_selfpromo.pt generated successfully!")
