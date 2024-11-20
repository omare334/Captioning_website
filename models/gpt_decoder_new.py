import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config

class CrossAttentionGPT2(nn.Module):
    def __init__(self, config, cross_attention_layers, num_blocks=10):
        super().__init__()
        self.gpt2 = GPT2Model(config)  # GPT2 model with all layers
        self.cross_attention_layers = cross_attention_layers  # List of cross-attention layers
        self.num_blocks = num_blocks  # Total number of blocks to use

    def forward(self, input_ids, attention_mask=None, image_embeddings=None):
        # Pass input through GPT-2's embedding layer (this applies token embedding)
        gpt2_outputs = self.gpt2(input_ids, attention_mask=attention_mask)
        
        # Extract the word embeddings (last hidden state)
        word_embeddings = gpt2_outputs.last_hidden_state  # Make sure you're using the right attribute
        
        # Process through alternating GPT-2 block and cross-attention layers
        for i in range(self.num_blocks):
            # Step 1: Apply GPT-2 attention block
            gpt2_block_output = self.gpt2.h[i](word_embeddings, attention_mask=attention_mask)
            word_embeddings = gpt2_block_output[0]  # Extract the last_hidden_state from the tuple
            
            # Print to show that we have gone through one GPT-2 block
            print(f"Passed through GPT-2 block {i + 1}, word embeddings shape: {word_embeddings.shape}")
            
            # Step 2: If image embeddings are provided, apply a different cross-attention for each block
            if image_embeddings is not None:
                # Use a different cross-attention layer for each block
                cross_attention_layer = self.cross_attention_layers[i]
                word_embeddings = cross_attention_layer(word_embeddings, image_embeddings)
                
                # Print to show that we have gone through cross-attention layer
                print(f"Passed through cross-attention after GPT-2 block {i + 1}, word embeddings shape: {word_embeddings.shape}")
        
        return word_embeddings

# Define your cross-attention layer (using your CrossAttention implementation)
class CrossAttention(nn.Module):
    def __init__(self, Wemb_dim, Pemb_dim, new_dim, num_heads, hidden_dim_ff, voc_size):
        super().__init__()
        self.num_heads = num_heads
        self.Whead_dim = new_dim // num_heads  # Embedding dimension for words per head
        self.Phead_dim = new_dim // num_heads  # Embedding dimension for images per head
        assert Wemb_dim % num_heads == 0, "Embedding dimension must be divisible by the number of heads"
        assert Pemb_dim % num_heads == 0, "Embedding dimension for images must be divisible by the number of heads"
        self.embeddings = nn.Embedding(num_embeddings=voc_size, embedding_dim=Wemb_dim)

        # Linear layers for query, key, and value transformations
        self.linear_q = nn.Linear(Wemb_dim, new_dim)
        self.linear_k = nn.Linear(Pemb_dim, new_dim)
        self.linear_v = nn.Linear(Pemb_dim, new_dim)

        # Linear layer for the concatenated output
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

# Create the custom GPT2 model
custom_config = GPT2Config(
    vocab_size=50300,
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

# Create a list of CrossAttention layers
num_blocks = 10
cross_attention_layers = [CrossAttention(768, 128, 128, 4, 6000, 50300) for _ in range(num_blocks)]

# Instantiate the model with the list of CrossAttention layers
model = CrossAttentionGPT2(custom_config, cross_attention_layers)

# Example input
input_ids = torch.randint(0, 50300, (2, 10))  # Dummy input with shape (batch_size, seq_len)
image_embeddings = torch.rand(2, 5, 128)  # Dummy image embeddings (batch_size, seq_len, emb_dim)

# Forward pass
output = model(input_ids, image_embeddings=image_embeddings)
print(output.shape)

