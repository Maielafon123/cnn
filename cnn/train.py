from model import Net
from dataset import train_loader
import torch.nn as nn
import torch




model = Net()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# print(batch_x.shape)
# print(batch_y.shape)
# print(logits.shape)
total_loss = 0


for epoch in range(5):
    model.train()
    total_loss = 0

    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()

        logits = model(batch_x)
        loss = criterion(logits, batch_y)

        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/5 | Loss: {avg_loss:.4f}")

# print(model.conv1.weight.grad)

