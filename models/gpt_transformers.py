import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config
from encoder import Encoder

class CrossAttention(nn.Module):
    def __init__(self, Wemb_dim, Pemb_dim, new_dim, num_heads, hidden_dim_ff, voc_size):
        super().__init__()
        self.num_heads = num_heads
        self.Whead_dim = new_dim // num_heads
        self.Phead_dim = new_dim // num_heads
        assert Wemb_dim % num_heads == 0, "Wemb_dim must be divisible by the number of heads"
        assert Pemb_dim % num_heads == 0, "Pemb_dim must be divisible by the number of heads"
        self.embeddings = nn.Embedding(num_embeddings=voc_size, embedding_dim=Wemb_dim)

        self.linear_q = nn.Linear(Wemb_dim, new_dim)
        self.linear_k = nn.Linear(Pemb_dim, new_dim)
        self.linear_v = nn.Linear(Pemb_dim, new_dim)

        self.linear_concat = nn.Linear(new_dim, Wemb_dim)
        self.norm = nn.LayerNorm(Wemb_dim)

    def forward(self, wemb, pemb):
        batch_size = wemb.size(0)
        Wseq_len = wemb.size(1)
        Pseq_len = pemb.size(1)

        query = self.linear_q(wemb).view(batch_size, Wseq_len, self.num_heads, self.Whead_dim).transpose(1, 2)
        key = self.linear_k(pemb).view(batch_size, Pseq_len, self.num_heads, self.Phead_dim).transpose(1, 2)
        value = self.linear_v(pemb).view(batch_size, Pseq_len, self.num_heads, self.Phead_dim).transpose(1, 2)

        scaling_factor = self.Whead_dim ** 0.5
        attention = torch.matmul(query, key.transpose(-2, -1)) / scaling_factor
        soft_matrix = torch.softmax(attention, dim=-1)
        sim_mat = soft_matrix @ value
        sim_mat = sim_mat.transpose(1, 2).contiguous()
        final_emb = sim_mat.view(batch_size, Wseq_len, -1)

        final_emb = self.linear_concat(final_emb)
        final_emb = self.norm(final_emb + wemb)
        return final_emb


class Transformer(nn.Module):
    def __init__(self, config, num_blocks, num_heads, voc_size, encoder_input_dim, encoder_output_dim, cross_att_dim):
        super().__init__()
        # GPT-2 backbone
        self.gpt2 = GPT2Model(config)

        # Encoder layers
        self.encoder_layers = nn.ModuleList(
            Encoder_layers = [Encoder(256, 256, 4, 4) for _ in range(num_blocks)]
        )

        # Cross-attention layers
        self.cross_attention_layers = nn.ModuleList(
            [CrossAttention(Wemb_dim=config.n_embd, Pemb_dim=encoder_output_dim, 
                            new_dim=cross_att_dim, num_heads=num_heads, 
                            hidden_dim_ff=64, voc_size=voc_size)
             for _ in range(num_blocks)]
        )

    def forward(self, input_ids, attention_mask=None, image_embeddings=None):
        # Pass through GPT-2's embedding layers
        gpt2_outputs = self.gpt2(input_ids, attention_mask=attention_mask)
        word_embeddings = gpt2_outputs.last_hidden_state

        # Process through encoder and cross-attention layers
        for i, (encoder, cross_attention) in enumerate(zip(self.encoder_layers, self.cross_attention_layers)):
            # Encoder processes image embeddings
            if image_embeddings is not None:
                image_embeddings = encoder(image_embeddings)
            
            # Cross-attention combines text and image embeddings
            if image_embeddings is not None:
                word_embeddings = cross_attention(word_embeddings, image_embeddings)

        return word_embeddings


# Example configuration and instantiation
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
transformer_model = Transformer(
    config=custom_config,
    num_blocks=num_blocks,
    num_heads=4,
    voc_size=50257,
    encoder_input_dim=256,
    encoder_output_dim=256,
    cross_att_dim=128
)

# Example input
input_ids = torch.randint(0, 50257, (2, 10))  # Batch size 2, sequence length 10
image_embeddings = torch.rand(2, 600, 256)  # Batch size 2, sequence length 600, embedding dim 256

# Forward pass
output = transformer_model(input_ids, attention_mask=None, image_embeddings=image_embeddings)
print("Output shape:", output.shape)
