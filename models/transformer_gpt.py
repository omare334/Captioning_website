import torch
import numpy as np

from models.encoder import Encoder
from models.GPT_decoder import Decoder2, CrossAttention, getPositionEncoding
from transformers import GPT2Model, GPT2Config

custom_config = GPT2Config(
    vocab_size=50297,
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
        num_encoder_layers=10,
        num_decoder_layers=10,
        gpt2_config=custom_config,
    ):
        super().__init__()
        # Image embedding layer
        self.img_embedding = torch.nn.Linear(pxl_size, emb_dim)
        
        # Word embedding layer
        self.word_embedding = torch.nn.Embedding(num_embeddings=voc_size, embedding_dim=Wemb_dim)
        
        # Encoder stack
        self.encoders = torch.nn.ModuleList(
            [Encoder(pxl_size, emb_dim, num_heads, hidden_dim_ff) for _ in range(num_encoder_layers)]
        )
        
        # Cross-attention layers (one for each decoder layer)
        self.cross_attention_layers = torch.nn.ModuleList(
            #Wemb_dim, Pemb_dim, new_dim, num_heads, hidden_dim_ff, voc_size
            [CrossAttention(Wemb_dim, Pemb_dim, new_dim, num_heads, hidden_dim_ff, voc_size) for _ in range(num_decoder_layers)]
        )
        
        # Decoder stack with Decoder2
        self.decoders = torch.nn.ModuleList(
            [Decoder2(gpt2_config, self.cross_attention_layers, num_blocks=num_decoder_layers)]
        )
        
        # Final projection layer
        self.project = torch.nn.Linear(Wemb_dim, voc_size)
    
    
    def forward(self, pxl_input, word_inpt, attention_mask=None):
        # Step 1: Pass through the encoder layers
        img_emb = self.img_embedding(pxl_input)  # Project pixel input to embedding space
        batch_size, seq_len, img_embed_dim = img_emb.size(0), img_emb.size(1), img_emb.size(2)
        
        # Add positional encoding to image embeddings
        sin_emb = getPositionEncoding(batch_size, seq_len, img_embed_dim)
        encoder_output = img_emb + sin_emb
        print("Initial Encoder Output Shape:", encoder_output.shape)
        
        for i, encoder in enumerate(self.encoders):
            encoder_output = encoder(encoder_output)
            print(f"Encoder Layer {i + 1} output shape:", encoder_output.shape)

        # Step 2: Pass through the word embedding layer
        wemb = self.word_embedding(word_inpt)  # Embed the word inputs
        batch_size, Wseq_len, Wd = wemb.size(0), wemb.size(1), wemb.size(2)

        # Add positional encoding to word embeddings
        Wsin_emb = getPositionEncoding(batch_size, Wseq_len, Wd)
        wemb = wemb + Wsin_emb
        print("Word Embeddings after Positional Encoding Shape:", wemb.shape)

        # Step 3: Pass through all decoder layers using Decoder2 logic
        for i, decoder in enumerate(self.decoders):
            # Decoder2 requires attention mask and encoder outputs (image embeddings) for cross-attention
            wemb = decoder(word_inpt=word_inpt, attention_mask=attention_mask, image_embeddings=encoder_output)
            print(f"Decoder Layer {i + 1} output shape:", wemb.shape)

        # Step 4: Project to vocabulary size
        prediction = self.project(wemb)  # Map embeddings to vocabulary logits
        print("Final Prediction Shape:", prediction.shape)
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