import torch
import numpy as np

from models.ViT import ViT
from models.decoder import Decoder2, MaskedAttention, CrossAttention, getPositionEncoding


# todo : 
# normalisation of layers
# batching 

class Transformer(torch.nn.Module):
    def __init__(
        self,
        pxl_size,
        emb_dim,
        num_heads,
        hidden_dim_ff,
        Wemb_dim,
        Pemb_dim,
        new_dim,
        voc_size,
        num_encoder_layers=6,
        num_decoder_layers=6,
    ):
        super().__init__()
        self.img_embedding = torch.nn.Linear(pxl_size,emb_dim)
        self.word_embedding = torch.nn.Embedding(num_embeddings=voc_size, embedding_dim=Wemb_dim)
        # Stacking encoder layers
        self.encoders = ViT()

        # Stacking decoder layers
        self.decoders = torch.nn.ModuleList(
            [Decoder2(voc_size, Wemb_dim, Pemb_dim, new_dim, num_heads, hidden_dim_ff) for _ in range(num_decoder_layers)]
        )

        self.project = torch.nn.Linear(Wemb_dim, voc_size)
    
    
    def forward(self, pxl_input, word_inpt):
        # Pass through all encoder layers
        # encoder_output = pxl_input
        #  = torch.nn.Linear(pxl_size,emb_dim)
        # encoder_output = self.linear(encoder_output)
        # img_emb = self.img_embedding(pxl_input)
        img_emb = pxl_input
        batch_size, _, seq_len, img_embed_dim = img_emb.size()
        # sin_emb = getPositionEncoding(batch_size, seq_len, img_embed_dim)
        # encoder_output = img_emb + sin_emb
        encoder_output = img_emb
        encoder_output = self.encoders(encoder_output)

        # Pass through all decoder layers
        wemb = self.word_embedding(word_inpt)
        # Positional encoding for word embeddings
        batch_size, Wseq_len, Wd = wemb.size(0), wemb.size(1), wemb.size(2)
        Wsin_emb = getPositionEncoding(batch_size, Wseq_len, Wd)
        print('The Wemb after adding positional encoding:', Wsin_emb.shape)
        wemb = wemb + Wsin_emb
        print('the output of wemb after embedding',wemb.shape)
        
        for i, decoder in enumerate(self.decoders):
            wemb = decoder(wemb, encoder_output)
            print(f"Decoder Layer {i + 1} output shape:", wemb.shape)

        prediction = self.project(wemb)

        return prediction



# omars test

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # Define dimensions
    # if emb_dim chnaged you need to change Pemb_dim aswell
    pxl_size = 784
    emb_dim = 256
    num_heads = 4
    hidden_dim_ff = 512
    Wemb_dim = 128
    Pemb_dim = 256
    new_dim = 256
    voc_size = 1000
    num_encoder_layers = 20
    num_decoder_layers = 20

    # Create model
    model = Transformer(
        pxl_size,
        emb_dim,
        num_heads,
        hidden_dim_ff,
        Wemb_dim,
        Pemb_dim,
        new_dim,
        voc_size,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
    )

    # # Dummy input data before batching
    # pxl_input = torch.rand((600, 784))  # Batch of 32 pixel inputs
    # print(pxl_input.shape)
    # wemb = torch.randint(0, voc_size, (32,))  # Batch of 32 word indices

    # Dummy input data
    pxl_input = torch.rand((2, 600, 784)).to(device)  # Batch of 32 pixel inputs
    print(pxl_input.shape)
    wemb = torch.rand(2, 600).to(device) # Batch of 32 word indices

    # Forward pass
    output = model(pxl_input, wemb)
    print("Final Transformer output shape:", output)