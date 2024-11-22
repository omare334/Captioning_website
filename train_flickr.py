from pathlib import Path
import sys
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
import torch
import torchvision
import os 

repo_dir = Path(__file__).parent.parent
sys.path.append(str(repo_dir))

from data.data_loader import Flickr
from models.transformer import Transformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_save_path = "transformer_model.pth"

transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(
            256
        ),  # Resize shorter side to 256 and keep aspect ratio
        torchvision.transforms.CenterCrop(256),  # Optionally crop the center to 256x256
        torchvision.transforms.ToTensor(),
    ]
)

train_dataset = Flickr("train", num_rows=10000, transform=transform)
# Create DataLoader with the custom collate function
train_loader = DataLoader(
    train_dataset, batch_size=64, shuffle=True, collate_fn=Flickr.collate_fn
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Available device is {device}")

# Save path for the model
model_save_path = "transformer_model.pth"

# Initialize the model
model = Transformer(256, 128, 8, 400, 128, 128, 64, 50300, 12, 12)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
criterion = torch.nn.CrossEntropyLoss()

wandb.init(project="image-captioning", name="none_gpt_200")

running_loss = []
running_accuracy = []

# Wrap training in a try-finally block
try:
    for epoch in range(200):
        for i, (patches, tokens, target, cap_lens) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch+1}")
        ):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            patches = patches.to(device)
            tokens = tokens.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            pred = model(patches, tokens)
            pred = torch.cat([x[: cap_lens[i]] for i, x in enumerate(pred)], dim=0)

            loss = criterion(pred.view(-1, pred.size(-1)), target.view(-1))
            loss.backward()
            optimizer.step()
            running_loss.append(loss.item())

            correct = (
                (torch.argmax(pred.view(-1, pred.size(-1)), dim=1) == target.view(-1))
                .sum()
                .item()
            )
            total = target.view(-1).size(0)
            accuracy = correct / total * 100
            running_accuracy.append(accuracy)

        # print("", end="\r")
        # print(f"loss: {sum(running_loss) / 100}", end="\r")
        # if (i+1) % 100 == 0:
        wandb.log(
            {
                "loss-100": sum(running_loss) / 100,
                "accuracy-100": sum(running_accuracy) / 100,
            }
        )
        running_loss = []
        running_accuracy = []