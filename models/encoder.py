from datasets import load_dataset
import torch
import numpy as np

# loaded_ds = torch.load('/Users/lydiafarnham/visual studio code/mlx5/final_model/flickr30k_patches.pt')


# encoder no masking . decoder cross attention no masking, separate class. only masking on the text - before cross attention, another class. no need for additional classes for encoder and decoder

def getPositionEncoding(seq_len, d, n=10000):
    P = np.zeros((seq_len, d))
    for k in range(seq_len):
        for i in np.arange(int(d/2)):
            denominator = np.power(n, 2*i/d)
            P[k, 2*i] = np.sin(k / denominator)
            P[k, 2*i+1] = np.cos(k / denominator)
    return torch.tensor(P, dtype=torch.float32)

class Encoder(torch.nn.Module):
    def __init__(self, pxl_size, emb_dim, num_heads, hidden_dim_ff):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads  # Dimension per head
        # print(self.head_dim)
        assert emb_dim % num_heads == 0, "Embedding dimension must be divisible by the number of heads"
        
        self.linear_q = torch.nn.Linear(emb_dim, emb_dim)
        self.linear_k = torch.nn.Linear(emb_dim, emb_dim)
        self.linear_v = torch.nn.Linear(emb_dim, emb_dim)
        
        # Learnable bias for attention
        self.attn_embedding_bias = torch.nn.Parameter(torch.zeros(emb_dim))

        self.linear_concat = torch.nn.Linear(emb_dim, emb_dim)

        self.norm = torch.nn.LayerNorm(emb_dim)
        
        # Feedforward layer (two linear layers with ReLU in between)
        self.feedforward = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, hidden_dim_ff),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim_ff, emb_dim)
        )

    def forward(self, emb):
        batch_size = emb.size(0)
        num_patches = emb.size(1)
        
        # Transform embeddings for query, key, and value
        query = self.linear_q(emb).view(batch_size, num_patches, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.linear_k(emb).view(batch_size, num_patches, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.linear_v(emb).view(batch_size, num_patches, self.num_heads, self.head_dim).transpose(1, 2)

        # Calculate attention scores and apply softmax
        scaling_factor = self.head_dim ** 0.5
        similarity_matrix = torch.matmul(query, key.transpose(-2, -1)) / scaling_factor

        # Apply softmax to get attention weights
        soft_matrix = torch.softmax(similarity_matrix, dim=-1)
    
        # Apply attention weights to values and reshape back
        attention = torch.matmul(soft_matrix, value)
        attention = attention.transpose(1, 2).contiguous()
        attention = attention.view(batch_size, num_patches, -1)  # recombine heads

        attention = self.linear_concat(attention)

        attention = self.norm(attention + emb)

        # Apply feedforward layer
        output = self.feedforward(attention)
        
        return output


if __name__ == '__main__':
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    embedding_dim = 64 # Set the embedding dimension
    # embedding_layer = torch.nn.Embedding(64, embedding_dim).to(device)

    args = (768, 128, 4, 4)  # Example arguments (vocab size, embedding size, num heads)
    model = Encoder(*args)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    # Random input tensor with integer values within embedding vocab range
    pemb = torch.rand((2,600,128)).to(device)  # Integer values for embedding lookup

    print(pemb.shape)
    output = model(pemb)
    print("Output shape:", output.shape)


    # for idx in range(len(loaded_ds)):
    #     patches = loaded_ds[idx]['patches']  # Get the patches for the current image
    #     unrolled_patches = []

        
    #     # Unroll each patch into a 1D vector
    #     for patch in patches:
    #         if isinstance(patch, list):
    #             print(patch)
    #             patch = np.array(patch)  # Convert to NumPy array

    #         # Flatten the patch and convert to a tensor
    #         unrolled_patch = torch.tensor(patch.flatten(), dtype=torch.float32).to(device)
    #         unrolled_patches.append(unrolled_patch)

    #     # Stack the unrolled patches into a single tensor
    #     unrolled_patches_tensor = torch.stack(unrolled_patches) 
    #     print(unrolled_patches_tensor.shape) # Shape: [num_patches, patch_size*patch_size*channels]

    #     output = model(unrolled_patches_tensor)  # Model output, expected shape [seq_len, vocab_size]
            
    #     print(f"Output shape for image {idx}: {output.shape}")

    # # Save the model's state dict if needed
    # torch.save(model.state_dict(), 'model_overfit.pth')