import torch
from Models.Transformer_Simple import Transformer_Simple
import torch.nn.functional as F

def main():
    # Hyperparameters
    src_pad_idx = 0
    trg_pad_idx = 10
    enc_voc_size = 5000  # Vocabulary size for the encoder
    dec_voc_size = 5000  # Vocabulary size for the decoder
    max_len = 50         # Maximum sequence length
    d_model = 512        # Model dimension
    n_head = 8           # Number of attention heads
    d_ff = 2048          # Feed-forward network dimension
    n_layers = 6         # Number of encoder/decoder layers
    activation = 'relu'  # Activation function
    dropout_p = 0.1      # Dropout probability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create the Transformer model
    model = Transformer_Simple(
        src_pad_idx=src_pad_idx,
        trg_pad_idx=trg_pad_idx,
        enc_voc_size=enc_voc_size,
        dec_voc_size=dec_voc_size,
        max_len=max_len,
        d_model=d_model,
        n_head=n_head,
        d_ff=d_ff,
        n_layers=n_layers,
        activation=activation,
        dropout_p=dropout_p,
        device=device
    ).to(device)

    # Dummy input data
    batch_size = 32  # Batch size
    src_seq_len = 40  # Source sequence length
    trg_seq_len = 30  # Target sequence length

    # Random source and target sequences (with padding index 0)
    src = torch.randint(1, enc_voc_size, (batch_size, src_seq_len), device=device)
    trg = torch.randint(1, dec_voc_size, (batch_size, trg_seq_len), device=device)

    src=F.pad(src, (0, src_pad_idx))
    trg=F.pad(trg, (0, trg_pad_idx))

    src = src.long()
    trg = trg.long()


    # Add padding tokens (simulating real data with padding)
    src[:, -5:] = src_pad_idx  # Padding last 5 tokens in source
    trg[:, -5:] = trg_pad_idx  # Padding last 5 tokens in target


    # Forward pass
    output = model(src, trg)

    # Output shape should be [batch_size, trg_seq_len, dec_voc_size]
    print(f"Output shape: {output.shape}")

if __name__ == "__main__":
    main()
