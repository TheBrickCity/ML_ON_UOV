import torch
import torch.nn as nn


class Gate(nn.Module):
    # rebuild of metas gate a small MLP (dim -> 4*dim -> dim) followed by a sigmoid, so every feature gets its own g in (0, 1)
    def __init__(self, dimension):
        super().__init__()
        self.fc1 = nn.Linear(dimension, 4 * dimension)
        self.fc2 = nn.Linear(4 * dimension, dimension)

    def forward(self, x):
        return torch.sigmoid(self.fc2(torch.relu(self.fc1(x))))


class TransformerLayer(nn.Module):
    # one layer; becomes a decoder layer (adds cross-attention) when encoder_dimension is given
    def __init__(self, dimension, num_heads, encoder_dimension=None, gated=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dimension, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dimension, eps=1e-12)

        self.cross_attn = None
        if encoder_dimension is not None:
            # keys and values are projected from the encoder's width straight into this layer's width,
            # so no separate encoder -> decoder projection layer is needed
            self.cross_attn = nn.MultiheadAttention(dimension, num_heads, kdim=encoder_dimension,
                                                    vdim=encoder_dimension, batch_first=True)
            self.norm_cross = nn.LayerNorm(dimension, eps=1e-12)

        self.ffn = nn.Sequential(nn.Linear(dimension, 4 * dimension), nn.ReLU(), nn.Linear(4 * dimension, dimension))
        self.norm2 = nn.LayerNorm(dimension, eps=1e-12)

        self.gate = Gate(dimension) if gated else None

    def forward(self, x, memory=None, causal_mask=None, loops=1):
        for _ in range(loops):  # SAME weights are reused on every loop (universal transformer)
            h = self.norm1(x + self.self_attn(x, x, x, attn_mask=causal_mask, need_weights=False)[0])
            g = self.gate(h) if self.gate is not None else None  # gate looks at the state after self-attention
            if self.cross_attn is not None:
                h = self.norm_cross(h + self.cross_attn(h, memory, memory, need_weights=False)[0])
            h = self.norm2(h + self.ffn(h))
            x = h if g is None else g * h + (1 - g) * x  # gated: blend new state with the old one
        return x


class TransformerStack(nn.Module):
    # embeddings + a stack of layers; the LAST layer is the gated, looped one. Used for both encoder and decoder.
    def __init__(self, vocab_size, max_len, dimension, num_heads, num_layers, loops, encoder_dimension=None):
        super().__init__()
        self.is_decoder = encoder_dimension is not None
        self.token_embed = nn.Embedding(vocab_size, dimension)
        self.position_embed = nn.Embedding(max_len, dimension)
        nn.init.normal_(self.token_embed.weight, mean=0, std=dimension ** -0.5)
        nn.init.normal_(self.position_embed.weight, mean=0, std=dimension ** -0.5)
        self.embed_norm = nn.LayerNorm(dimension, eps=1e-12)
        self.layers = nn.ModuleList([
            TransformerLayer(dimension, num_heads, encoder_dimension, gated=(i == num_layers - 1))
            for i in range(num_layers)
        ])
        self.loops = loops

    def forward(self, tokens, memory=None):
        batch, seq_length = tokens.shape
        positions = torch.arange(seq_length, device=tokens.device).unsqueeze(0)
        x = self.embed_norm(self.token_embed(tokens) + self.position_embed(positions))

        causal_mask = None
        if self.is_decoder:  # decoder positions may only look at earlier positions
            causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_length, device=tokens.device)

        for i, layer in enumerate(self.layers):
            loops = self.loops if i == len(self.layers) - 1 else 1
            x = layer(x, memory=memory, causal_mask=causal_mask, loops=loops)
        return x


class UOVTransformer(nn.Module):
    # default parameters mirror SALSA defaults as implemented by facebookresearch
    def __init__(self, vocab_size, max_src_len, max_tgt_len, enc_dim=1024, dec_dim=512, enc_heads=16, dec_heads=4,
                 enc_layers=2, dec_layers=2, enc_loops=2, dec_loops=8):
        super().__init__()
        self.encoder = TransformerStack(vocab_size, max_src_len, enc_dim, enc_heads, enc_layers, enc_loops)
        self.decoder = TransformerStack(vocab_size, max_tgt_len, dec_dim, dec_heads, dec_layers, dec_loops,
                                        encoder_dimension=enc_dim)
        self.output_layer = nn.Linear(dec_dim, vocab_size)

    def forward(self, src, tgt_in):
        # src: (batch, src_len), tgt_in: (batch, k) -> logits (batch, k, vocab_size)
        memory = self.encoder(src)
        return self.output_layer(self.decoder(tgt_in, memory))

    @torch.no_grad()
    def generate(self, src, sos_token, tgt_len):
        # greedy decoding; returns (batch, tgt_len) tokens starting with SOS, same layout as dataset.tgt
        was_training = self.training
        self.eval()
        memory = self.encoder(src)
        generated = torch.full((src.shape[0], 1), sos_token, dtype=torch.long, device=src.device)
        for _ in range(tgt_len - 1):
            logits = self.output_layer(self.decoder(generated, memory))
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
        self.train(was_training)
        return generated