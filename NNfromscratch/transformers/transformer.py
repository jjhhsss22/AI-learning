import torch
import torch.nn as nn


class selfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(selfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert(self.head_dim * self.heads == embed_size),"embed_size must be divisible by heads"


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

        out.reshape(N, query_len, self.heads * self.head_dim)
        out = self.fc_out(out)  # concatenate all the final contextualised sums from each head into a final big embedding for each token in a batch