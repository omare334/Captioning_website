from datasets import load_dataset
import torch
import torchvision
import random
import PIL
from pathlib import Path
from torch.utils.data import Subset
from tqdm import tqdm
import os
import pickle
import sentencepiece as spm

path_to_split_ind = Path(__file__).parent.parent / "data/split_to_indices.pkl"
path_to_tokeniser = Path(__file__).parent.parent / "final_model/tokenizer.model"


class Flickr(torch.utils.data.Dataset):
    def __init__(
        self, split: str, num_rows: int, window_size: int = 16, transform=None
    ):
        super().__init__()
        self.window_size = window_size
        self.ds = load_dataset("nlphuji/flickr30k")
        self.tokeniser = spm.SentencePieceProcessor(model_file=str(path_to_tokeniser))
        self.vocab_size = self.tokeniser.get_piece_size()
        self.split = split
        if os.path.exists(path_to_split_ind):
            with open(path_to_split_ind, "rb") as f:
                split_to_indices = pickle.load(f)
        else:
            split_to_indices = {"train": [], "test": [], "val": []}
            for idx, row in enumerate(tqdm(self.ds["test"], desc="Getting Splits")):
                split_to_indices[row["split"]].append(idx)

            print(f"Saving split indices to {path_to_split_ind}")
            with open(path_to_split_ind, "wb") as f:
                pickle.dump(split_to_indices, f)

        if num_rows != -1:
            self.split_indices = split_to_indices[split][:num_rows]
        else:
            self.split_indices = split_to_indices[split]
        self.transform = transform

    def __len__(self):
        return len(self.split_indices)

    def __getitem__(self, idx):
        img = self.ds["test"][self.split_indices[idx]]["image"]
        if self.transform:
            img = self.transform(img)
        captions = self.ds["test"][self.split_indices[idx]]["caption"]
        caption = captions[0]

        patches = self.get_patches(img)
        caption_tokens = self.encode_label(caption)
        targ = caption_tokens[1:]
        inpt = caption_tokens[:-1].clone().detach()
        return (
            patches,
            inpt,
            targ,
        )

    def collate_fn(batch):
        patches, tokens, targets = zip(*batch)
        cap_lens = [len(string) for string in tokens]
        padded_inpts = torch.nn.utils.rnn.pad_sequence(
            tokens, batch_first=True, padding_value=0
        )

        # Concatenate images and labels
        targets = torch.cat(targets, dim=0)
        images = torch.cat(patches, dim=0)

        return images, padded_inpts, targets, cap_lens

    def encode_label(self, label):
        return torch.LongTensor(
            [self.tokeniser.bos_id()]
            + self.tokeniser.encode(label)
            + [self.tokeniser.eos_id()]
        )

    def get_patches(self, img: torch.Tensor):
        img = img.unsqueeze(0)

        _, num_channels, height, width = img.shape

        # Check if image dimensions are divisible by window_size
        assert (
            height % self.window_size == 0 and width % self.window_size == 0
        ), "Height and width must be divisible by the window size."

        # Use unfold to extract patches
        windows = img.unfold(2, self.window_size, self.window_size).unfold(
            3, self.window_size, self.window_size
        )

        # patches = torch.cat([patches_r, patches_g, patches_b], dim=0)
        # Shape: (1, num_patches_h, num_patches_w, window_size, window_size)

        patches = windows.reshape(1, -1, self.window_size * self.window_size)
        # Shape: (1, num_windows, channels * window_size * window_size)

        return patches


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(
                256
            ),  # Resize shorter side to 256 and keep aspect ratio
            torchvision.transforms.CenterCrop(
                256
            ),  # Optionally crop the center to 256x256
            torchvision.transforms.ToTensor(),
        ]
    )
    ds = Flickr("train",-1, transform=transform)
    train_loader = DataLoader(
        ds, batch_size=32, shuffle=True, collate_fn=Flickr.collate_fn
    )
    for x in train_loader:
        pass
        print(x)