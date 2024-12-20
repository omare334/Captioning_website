from pathlib import Path
import sys
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
import torch
import torchvision
from models.encoder import Encoder
from models.gpt_decoder_new import CrossAttention,Transformer
from transformers import GPT2Config

repo_dir = Path(__file__).parent.parent
sys.path.append(str(repo_dir))

from data.data_loader import Flickr

transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(
            256
        ),  # Resize shorter side to 256 and keep aspect ratio
        torchvision.transforms.CenterCrop(256),  # Optionally crop the center to 256x256
        torchvision.transforms.ToTensor(),
    ]
)

train_dataset = Flickr("train", num_rows=100, transform=transform)
# Create DataLoader with the custom collate function
train_loader = DataLoader(
    train_dataset, batch_size=64, shuffle=True, collate_fn=Flickr.collate_fn
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Available device is {device}")

custom_config = GPT2Config(
    vocab_size=50257,
    n_positions=1024,
    n_ctx=1024,
    n_embd=768,
    n_head=12,
    n_layer=12,
    resid_pdrop=0.1,
    attn_pdrop=0.1,
    embd_pdrop=0.1,
    use_cache=True
)

num_blocks = 10
encoder_layers = [Encoder(756, 756, 4, 4) for _ in range(num_blocks)]
cross_attention_layers = [
    CrossAttention(Wemb_dim=756, Pemb_dim=756, new_dim=128, num_heads=4, hidden_dim_ff=64, voc_size=50257)
    for _ in range(num_blocks)
]

#'pxl_size', 'emb_dim', 'num_heads', 'hidden_dim_ff', 'Wemb_dim', 'Pemb_dim', 'new_dim', and 'voc_size'
model = Transformer(custom_config, cross_attention_layers, encoder_layers, num_blocks)

model.to(device)

print(
    f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
criterion = torch.nn.CrossEntropyLoss()

wandb.init(project="image-captioning", name="flickr-multi-head_GELU_100")
running_loss = []
running_accuracy = []
for epoch in range(1):
    for i, (patches, tokens, target, cap_lens) in enumerate(
        tqdm(train_loader, desc="Training")
    ):
        patches = patches.to(device)
        print(patches.shape)
        tokens = tokens.to(device)
        print(tokens.shape)
        target = target.to(device)

        optimizer.zero_grad()
        pred = model(tokens,None, patches)
        pred = torch.cat([x[: cap_lens[i]] for i, x in enumerate(pred)], dim=0)

        loss = criterion(pred.view(-1, pred.size(-1)), target.view(-1))
        loss.backward()
        optimizer.step()
        running_loss.append(loss.item())

        correct = (
            (torch.argmax(pred.view(-1, pred.size(-1)), dim=1) == target.view(-1))
            .sum()
            .item()
        )  # Count correct predictions
        total = target.view(-1).size(0)  # Total number of predictions
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
