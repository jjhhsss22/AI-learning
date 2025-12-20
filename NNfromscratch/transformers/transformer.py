import torch
import torch.nn as nn
from torch.testing._internal.distributed._tensor.common_dtensor import FeedForward


class selfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(selfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert(self.head_dim * self.heads == embed_size),"embed_size must be divisible by heads"


        '''
        Q K V are calculated on every forward pass for each embedding (token)
        by applying three learned linear projections to each input token embedding
        these weights will change across batches as they are updated during backpropagation 
        '''

        # values = what information is stored in each token (its significance varies to different tokens depending on its attention score)
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)  # original paper doesn't use bias - all it does is complicate the maths

        # keys = how each token presents itself to tokens (including itself)
        self.key = nn.Linear(self.head_dim, self.head_dim, bias=False)

        # queries = how much does each token (including itself) matter to the CURRENT token
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)

        # concatenates all outputs of each head back to the dimensions of the embeddings
        self.fc_out = nn.Linear(self.heads*self.head_dim, self.embed_size)




    def forward(self, values, key, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], key.shape[1], query.shape[1]

        # splitting embeddings into individual pieces for the heads (splits the full embedding into multiple parts)
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = key.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        energy = torch.einsum('nqhd,nkhd->nhqk', [queries, keys])
        '''
        n	batch
        q	query position
        k	key position
        h	head
        d	feature dimension (inside 1 head)
        
        this torch.einsum is computing:
            the Q × Kᵀ
        for each batch, each head, each query token, each key token,
        it performs dot product
        
            for each (n, h, q, k):
                sum over d → queries[n,q,h,d] * keys[n,k,h,d]        
            
        the final energy tensor stores the attention scores for
        ALL batches, ALL heads, and ALL query–key pairs.
        
        energy shape = (batch, heads, query_len, key_len)
        '''


        if mask is not None:
            '''
            turn values that need to be masked to -inf.
            so that when softmax is applied on them, it will return 0.
            
            for look-ahead masking
            so that decoder can't see future tokens and attempt to make predictions instead
            or
            padding tokens
            '''
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        scaled_energy = energy / (self.head_dim ** 0.5)
        '''
        dot product grows with dimension
        so softmax can become 'peaky' and gradients will vanish
        in the original paper, scaling resolves this issue by keeping dot products in a reasonable range 
        '''

        attention = torch.softmax(scaled_energy, dim=3)  # dim=3 represents the dim of keys so performs softmax for each query
        # now each row sums up to 1 -> Σ attention[n,h,q,k] = 1 (for each batch, each head, each query, distribute attention across keys)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values])
        '''
        this einsum operation multiplies attention weights with their corresponding value vectors,
        producing a weighted sum over ALL keys for EACH query token and EACH head.
        
        the result is a contextualised embedding where each head captures different relationships in the sequence.
        
        it is computing:
            the Σ over k ( V x softmax(attention_qv) )
        
        out[n, q, h, d] =
            Σ over k ( softmax(QKᵀ / √dₖ) · V )
        
        attention.shape = (N, heads, query_len, key_len)
        values.shape    = (N, key_len, heads, head_dim)
        out.shape = (N, query_len, heads, head_dim)  - final sum output for each head for each sequence in a batch
        '''

        # concatenate the final contextualised output from all attention heads
        # into a final big embedding for each token in a batch
        out = out.reshape(N, query_len, self.heads * self.head_dim)
        # apply a shared linear transformation (weights + bias) to each token embedding
        # to recombine information from all heads and project back to embed_size
        out = self.fc_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = selfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)  # layer norm normalises for every sequence across its tokens
        self.norm2 = nn.LayerNorm(embed_size)  # z = (x - μ) / σ

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )  # feed forward layer that has a bigger hidden layer - gives the model nonlinear capacity

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))  # add and normalise like the original paper
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))

        return out

class Encoder(nn.Module):
    def __init__(self,
                 src_vocab_size,
                 embed_size,
                 num_layers,
                 heads,
                 device,
                 forward_expansion,
                 dropout,
                 max_length
        ):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList([
            TransformerBlock(embed_size, heads, forward_expansion=forward_expansion, dropout=dropout)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)

        embed_w_time_signal = self.word_embedding(x) + self.position_embedding(positions)
        out = self.dropout(embed_w_time_signal)

        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out

# The encoder learns contextual representations; the decoder uses them as keys and values during cross-attention.
# Encoder - Understand the input sentence.
# Decoder - At each step, ask the encoder: what parts of the input matter for my next word?