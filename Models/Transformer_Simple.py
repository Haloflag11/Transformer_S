import torch
from torch import nn
import torch.nn.functional as F
from Layers.Encoder import Encoder
from Layers.Decoder import Decoder

class Transformer_Simple(nn.Module):
    def __init__(self,
                 src_pad_idx,
                 trg_pad_idx,
                 enc_voc_size,
                 dec_voc_size,
                 max_len,
                 d_model,
                 n_head,
                 d_ff,
                 n_layers,
                 activation='relu',
                 dropout_p=0.1,
                 device='cpu'
                 ):
        """
            Initializes the Transformer model constructor.

            :param src_pad_idx: int
                The index used for padding in the source sequences. This is used to mask out the padding tokens in source sequences for attention mechanisms.

            :param trg_pad_idx: int
                The index used for padding in the target sequences. Similar to src_pad_idx, it's used to mask out padding tokens in target sequences.

            :param enc_voc_size: int
                Vocabulary size of the encoder. Defines the total number of distinct tokens that the encoder can process.

            :param dec_voc_size: int
                Vocabulary size of the decoder. Defines the total number of distinct tokens that the decoder can generate.

            :param max_len: int
                The maximum length of the source sequences.
            :param d_model: int
                The dimensionality of the outputs for all layers in the model. This core parameter of the Transformer model affects the depth of the network.

            :param n_head: int
                The number of heads in the multi-head attention mechanisms. Each head offers a different perspective and focuses on different parts of the information, enhancing the learning capabilities of the model.

            :param d_ff: int
                The dimensionality of the feed-forward network's intermediate layer. Typically larger than `d_model`, it is used to increase the model’s expressiveness in the non-linear layer.

            :param n_layers: int
                The number of encoder and decoder layers stacked to form the Transformer model.

            :param activation: str or callable
                The type of activation function used. Common activation functions include 'relu', 'gelu', etc.

            :param dropout_p: float
                The dropout rate applied throughout the model to prevent overfitting.

            :param device: str, optional
                The device ('cpu' or 'gpu') on which the model and its tensors will be allocated. Defaults to 'cpu'.
            """
        super(Transformer_Simple,self).__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.activation = {

            'relu': F.relu,
            'gelu': F.gelu,
            'swish': F.silu,
        }
        self.activation_func = self.activation[activation]
        self.encoder=Encoder(enc_voc_size,max_len,d_model,d_ff,self.activation_func,n_head,n_layers,dropout_p,device)
        self.decoder=Decoder(dec_voc_size,max_len,d_model,n_head,d_ff,self.activation_func,n_layers,dropout_p,device)
        self.device=device

        assert activation in ['relu','gelu','swish'],f'Activation {activation} is not supported.'
        assert device in ['cpu','cuda'],'Device must be ''cpu'' or ''cuda'' '

    def make_pad_mask(self, q, k, pad_idx_q, pad_idx_k):
        """
           Generate padding masks for the attention mechanism.

           :param q: Query tensor, shape [B, T_q]
           :param k: Key tensor, shape [B, T_k]
           :param pad_idx_q: Padding index for the query
           :param pad_idx_k: Padding index for the key
           :return: Padding mask, shape [B, 1, len_q, len_k]
           """

        len_q, len_k = q.size(1), k.size(1)

        # Query mask: [B, T_q] -> [B, 1, T_q, 1]
        q_mask = q.ne(pad_idx_q).unsqueeze(1).unsqueeze(3)

        # Key mask: [B, T_k] -> [B, 1, 1, T_k]
        k_mask = k.ne(pad_idx_k).unsqueeze(1).unsqueeze(2)

        # Combine masks: [B, 1, T_q, T_k]
        mask = q_mask & k_mask
        return mask

    def make_casual_mask(self, q, k):
        """

           :param q: Tensor of shape [B, T_q, d_model]
               The query tensor. Used to determine the sequence length (T_q).
           :param k: Tensor of shape [B, T_k, d_model]
               The key tensor. Used to determine the sequence length (T_k).
           :return: Tensor of shape [1,T_q, T_k]

           """
        len_q, len_k = q.size(1), k.size(1) # Create a lower triangular matrix of shape [T_q, T_k]
        mask = torch.tril(torch.ones(len_q, len_k, device=self.device)).bool().unsqueeze(0)
        return mask

    def forward(self, src, trg):
        src_mask = self.make_pad_mask(src,src,self.src_pad_idx,self.src_pad_idx)
        trg_mask = self.make_pad_mask(trg,trg,self.trg_pad_idx,self.trg_pad_idx)&self.make_casual_mask(trg,trg)
        enc=self.encoder(src,src_mask)
        dec=self.decoder(trg,enc,src_mask,trg_mask)
        return dec

