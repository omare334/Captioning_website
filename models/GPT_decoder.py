import torch
import numpy as np
from transformers import GPT2Tokenizer
import torch as nn
import torch.nn as nn  # Import nn as alias for nn.Module

from transformers import GPT2Config, GPT2Model

<<<<<<< Updated upstream
=======
from gpt_decoder_new import CrossAttentionGPT2



>>>>>>> Stashed changes
def getPositionEncoding(batch_size, seq_len, d, n=10000):
    P = np.zeros((batch_size, seq_len, d))  # Adjusted to include batch size
    for k in range(seq_len):
        for i in np.arange(int(d/2)):
            denominator = np.power(n, 2*i/d)
            batch_indices = np.arange(batch_size)  # Shape (batch_size,)
            P[:, k, 2 * i] = np.sin(batch_indices / denominator)  # Broadcasting for batch size
            P[:, k, 2 * i + 1] = np.cos(batch_indices / denominator)
    return torch.tensor(P, dtype=torch.float32)  # Convert to PyTorch tensor

<<<<<<< Updated upstream
   
class Decoder2(nn.Module):
    def __init__(self, config, cross_attention_layers, num_blocks=10):
        super().__init__()
        self.gpt2 = GPT2Model(config)  # GPT2 model with all layers
        self.cross_attention_layers = cross_attention_layers  # List of cross-attention layers
        self.num_blocks = num_blocks  # Total number of blocks to use

    def forward(self, input_ids, image_embeddings, attention_mask=None):
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

=======

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

    
class Decoder2(torch.nn.Module):
    def __init__(self, vocab_size, Wemb_dim, Pemb_dim, new_dim, num_heads, hidden_dim_ff):
        super().__init__()
        self.masked_attn = MaskedAttention(Wemb_dim, num_heads, hidden_dim_ff)
        self.cross_attn = CrossAttentionGPT2(Wemb_dim, Pemb_dim, new_dim, num_heads, hidden_dim_ff, vocab_size)
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(Wemb_dim, hidden_dim_ff),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim_ff, Wemb_dim)
        )
>>>>>>> Stashed changes
    
class CrossAttention(torch.nn.Module):
    def __init__(self, Wemb_dim, Pemb_dim, new_dim, num_heads, hidden_dim_ff, voc_size):
        super().__init__()
        self.num_heads = num_heads
        self.Whead_dim = new_dim // num_heads  # Embedding dimension for words per head
        self.Phead_dim = new_dim // num_heads  # Embedding dimension for images per head

        assert Wemb_dim % num_heads == 0, "Embedding dimension must be divisible by the number of heads"
        assert Pemb_dim % num_heads == 0, "Embedding dimension for images must be divisible by the number of heads"

        self.embeddings = torch.nn.Embedding(num_embeddings=voc_size, embedding_dim=Wemb_dim)

        # Linear layers for query, key, and value transformations
        self.linear_q = torch.nn.Linear(Wemb_dim, new_dim)
        self.linear_k = torch.nn.Linear(Pemb_dim, new_dim)
        self.linear_v = torch.nn.Linear(Pemb_dim, new_dim)

        # Linear layer for the concatenated output
        self.linear_concat = torch.nn.Linear(new_dim, Wemb_dim)

        self.norm = torch.nn.LayerNorm(Wemb_dim)

    def forward(self, wemb, pemb):
        # wemb: [batch_size, seq_len_w, Wemb_dim]
        # pemb: [batch_size, seq_len_p, Pemb_dim]
        # batch_size,seq_len, wemb_dim
        # No positional encoding needed for image embeddings (Pemb)
        batch_size = wemb.size(0)
        batch_size = pemb.size(0)
        Wseq_len = wemb.size(1)
        Pseq_len = pemb.size(1)
        
     
        print("The Pemb shape:", pemb.shape) 

        # Transform embeddings for query, key, and value
        query = self.linear_q(wemb).view(batch_size, Wseq_len, self.num_heads, self.Whead_dim).transpose(1, 2)
        key = self.linear_k(pemb).view(batch_size, Pseq_len, self.num_heads, self.Phead_dim).transpose(1, 2)
        value = self.linear_v(pemb).view(batch_size, Pseq_len, self.num_heads, self.Phead_dim).transpose(1, 2)

        print("Query shape after linear transformation:", query.shape)
        print("Key shape after linear transformation:", key.shape)
        print("Value shape after linear transformation:", value.shape)

        # Attention computation: query * key^T
        scaling_factor = self.Whead_dim ** 0.5  # or use self.Phead_dim if necessary
        attention = torch.matmul(query, key.transpose(-2, -1)) / scaling_factor

        # Apply softmax to get attention weights
        soft_matrix = torch.softmax(attention, dim=-1)

        # Attention output
        # matmul replacement
        sim_mat = soft_matrix @ value  
        sim_mat = sim_mat.transpose(1, 2).contiguous()
        final_emb = sim_mat.view(batch_size, Wseq_len, -1)  # Reshape to (Wseq_len, num_heads * Whead_dim)

        # Pass through the linear layer after concatenation
        final_emb = self.linear_concat(final_emb)

        final_emb = self.norm(final_emb + wemb)
        # add residual and normalization

        return final_emb

if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Test parameters
    vocab_size = 6000  # Vocabulary size
    Wemb_dim = 64  # Word embedding dimension
    Pemb_dim = 128  # Positional embedding dimension
    new_dim = 128  # Projection dimension
    num_heads = 4  # Number of attention heads
    hidden_dim_ff = 256  # Hidden dimension for feed-forward
    num_blocks = 3  # Number of blocks (cross-attention + GPT layers)

    # Initialize cross-attention layers
    cross_attention_layers = [
        CrossAttention(Wemb_dim, Pemb_dim, new_dim, num_heads, hidden_dim_ff, vocab_size).to(device)
        for _ in range(num_blocks)
    ]

    # Create GPT-2 configuration
    config = GPT2Config(vocab_size=vocab_size, n_embd=Wemb_dim, n_layer=num_blocks, n_head=num_heads)

    # Initialize the Decoder2 model
    model = Decoder2(config, cross_attention_layers, num_blocks=num_blocks).to(device)

    # Optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    # Random input tensors
    batch_size = 2
    seq_len_w = 10  # Word sequence length
    seq_len_p = 600  # Positional sequence length

    pemb = torch.rand((2,600,128)).to(device)  # Integer values for embedding lookup
    wemb = torch.rand(2,10, 64).to(device)

    # Attention mask (optional)
    attention_mask = torch.ones(batch_size, seq_len_w).to(device)

    # Forward pass
    try:
        output = model(wemb, pemb,attention_mask)
        print(f"Output shape: {output.shape}")
    except Exception as e:
        print(f"Error during forward pass: {e}")
