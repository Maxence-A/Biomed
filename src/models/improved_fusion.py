"""
Improved Fusion Network for Protein Subcellular Localization

Architecture amelioree basee sur l'etat de l'art 2024-2025:
1. Bidirectional Cross-Attention: ESM <-> ProstT5 (interaction bidirectionnelle)
2. BiLSTM: Capture des dependances sequentielles (inspire de LocPro 2025)
3. Multi-Head Attention Pooling: Agregation adaptive multi-tete

References:
- LocPro 2025: ESM2 + BiLSTM (+10% F1)
- BioLangFusion: Cross-modal multi-head attention
- HEAL: Hierarchical attention pooling
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BidirectionalCrossAttention(nn.Module):
    """
    Cross-attention BIDIRECTIONNELLE entre ESM-2 et ProstT5

    Contrairement a la version simple:
    - ESM attend sur ProstT5 (sequence -> structure)
    - ProstT5 attend sur ESM (structure -> sequence)
    - Les deux sont combines

    Cela permet une vraie interaction entre les modalites.
    """

    def __init__(
        self,
        esm_dim: int = 1280,
        prost_dim: int = 1024,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Projections vers dimension commune
        self.esm_proj = nn.Linear(esm_dim, hidden_dim)
        self.prost_proj = nn.Linear(prost_dim, hidden_dim)

        # Cross-attention ESM -> ProstT5 (ESM query, ProstT5 key/value)
        self.cross_attn_esm_to_prost = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Cross-attention ProstT5 -> ESM (ProstT5 query, ESM key/value)
        self.cross_attn_prost_to_esm = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Fusion des deux directions
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=-1)
        )

        # Output projection avec residual
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, esm_emb, prost_emb, attention_mask=None):
        """
        Args:
            esm_emb: (batch, seq_len, 1280)
            prost_emb: (batch, seq_len, 1024)
            attention_mask: (batch, seq_len)

        Returns:
            fused: (batch, seq_len, hidden_dim)
        """
        # Projections
        esm_proj = self.esm_proj(esm_emb)      # (batch, seq, hidden)
        prost_proj = self.prost_proj(prost_emb)  # (batch, seq, hidden)

        # Key padding mask
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = (attention_mask == 0)

        # Direction 1: ESM attend sur ProstT5 (enrichit ESM avec info structurelle)
        esm_enriched, _ = self.cross_attn_esm_to_prost(
            query=esm_proj,
            key=prost_proj,
            value=prost_proj,
            key_padding_mask=key_padding_mask
        )
        esm_enriched = esm_enriched + esm_proj  # Residual

        # Direction 2: ProstT5 attend sur ESM (enrichit structure avec info sequence)
        prost_enriched, _ = self.cross_attn_prost_to_esm(
            query=prost_proj,
            key=esm_proj,
            value=esm_proj,
            key_padding_mask=key_padding_mask
        )
        prost_enriched = prost_enriched + prost_proj  # Residual

        # Fusion adaptive des deux directions
        concat = torch.cat([esm_enriched, prost_enriched], dim=-1)
        gate_weights = self.fusion_gate(concat)  # (batch, seq, 2)

        # Combinaison ponderee
        fused = (gate_weights[:, :, 0:1] * esm_enriched +
                 gate_weights[:, :, 1:2] * prost_enriched)

        # Output projection
        fused = self.output_proj(fused)

        return fused


class BiLSTMEncoder(nn.Module):
    """
    BiLSTM pour capturer les dependances sequentielles

    Inspire de LocPro 2025 qui a obtenu +10% F1 avec cette approche.
    Le BiLSTM capture les patterns locaux et globaux dans la sequence.
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Output dim = hidden_dim * 2 (bidirectionnel)
        self.output_dim = hidden_dim * 2

        # Layer norm pour stabiliser
        self.layer_norm = nn.LayerNorm(self.output_dim)

    def forward(self, x, attention_mask=None):
        """
        Args:
            x: (batch, seq_len, input_dim)
            attention_mask: (batch, seq_len) - optional

        Returns:
            output: (batch, seq_len, hidden_dim * 2)
        """
        # Pack si mask fourni (pour efficacite)
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1).cpu()
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            output, _ = self.lstm(packed)
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

            # Padding si necessaire
            if output.size(1) < x.size(1):
                padding = torch.zeros(
                    output.size(0),
                    x.size(1) - output.size(1),
                    output.size(2),
                    device=output.device
                )
                output = torch.cat([output, padding], dim=1)
        else:
            output, _ = self.lstm(x)

        output = self.layer_norm(output)
        return output


class MultiHeadAttentionPooling(nn.Module):
    """
    Attention Pooling multi-tete pour agregation

    Inspire de HEAL: utilise plusieurs tetes d'attention pour
    capturer differents aspects de la sequence.
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        # Query learnable (comme un CLS token)
        self.query = nn.Parameter(torch.randn(1, num_heads, self.head_dim))

        # Projections pour multi-head
        self.key_proj = nn.Linear(embedding_dim, embedding_dim)
        self.value_proj = nn.Linear(embedding_dim, embedding_dim)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.Dropout(dropout)
        )

        self.scale = math.sqrt(self.head_dim)

    def forward(self, embeddings, attention_mask=None):
        """
        Args:
            embeddings: (batch, seq_len, dim)
            attention_mask: (batch, seq_len)

        Returns:
            pooled: (batch, dim)
            weights: (batch, num_heads, seq_len)
        """
        batch_size, seq_len, dim = embeddings.shape

        # Projections
        keys = self.key_proj(embeddings)  # (batch, seq, dim)
        values = self.value_proj(embeddings)  # (batch, seq, dim)

        # Reshape pour multi-head
        keys = keys.view(batch_size, seq_len, self.num_heads, self.head_dim)
        keys = keys.permute(0, 2, 1, 3)  # (batch, heads, seq, head_dim)

        values = values.view(batch_size, seq_len, self.num_heads, self.head_dim)
        values = values.permute(0, 2, 1, 3)  # (batch, heads, seq, head_dim)

        # Query broadcast
        query = self.query.expand(batch_size, -1, -1)  # (batch, heads, head_dim)
        query = query.unsqueeze(2)  # (batch, heads, 1, head_dim)

        # Attention scores
        scores = torch.matmul(query, keys.transpose(-2, -1)) / self.scale
        scores = scores.squeeze(2)  # (batch, heads, seq)

        # Mask padding
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1)  # (batch, 1, seq)
            scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax
        weights = F.softmax(scores, dim=-1)  # (batch, heads, seq)

        # Weighted sum
        weights_expanded = weights.unsqueeze(-1)  # (batch, heads, seq, 1)
        pooled = (values * weights_expanded).sum(dim=2)  # (batch, heads, head_dim)

        # Concat heads
        pooled = pooled.view(batch_size, -1)  # (batch, dim)

        # Output projection
        pooled = self.output_proj(pooled)

        return pooled, weights


class ClassificationHead(nn.Module):
    """
    MLP classification head with LayerNorm and Dropout
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = [512, 256],
        num_classes: int = 6,
        dropout: float = 0.3
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        return self.classifier(x)


class ImprovedFusionNetwork(nn.Module):
    """
    Architecture amelioree pour localisation subcellulaire

    Pipeline:
    1. Bidirectional Cross-Attention (ESM <-> ProstT5)
    2. BiLSTM (dependances sequentielles)
    3. Multi-Head Attention Pooling (agregation adaptive)
    4. Classification Head

    Cette architecture:
    - Utilise UNIQUEMENT ESM-2 + ProstT5 (conforme au sujet)
    - Est DIFFERENTE de DeepLocPro (qui n'a pas BiLSTM ni cross-attention bidirectionnelle)
    - Combine les meilleures pratiques de LocPro, HEAL, BioLangFusion
    """

    def __init__(
        self,
        esm_dim: int = 1280,
        prost_dim: int = 1024,
        hidden_dim: int = 512,
        lstm_hidden: int = 256,
        num_heads: int = 8,
        num_lstm_layers: int = 2,
        num_classes: int = 6,
        dropout: float = 0.3
    ):
        super().__init__()

        # 1. Bidirectional Cross-Attention Fusion
        self.cross_attention = BidirectionalCrossAttention(
            esm_dim=esm_dim,
            prost_dim=prost_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        # 2. BiLSTM Encoder
        self.bilstm = BiLSTMEncoder(
            input_dim=hidden_dim,
            hidden_dim=lstm_hidden,
            num_layers=num_lstm_layers,
            dropout=dropout
        )

        # 3. Multi-Head Attention Pooling
        self.attention_pooling = MultiHeadAttentionPooling(
            embedding_dim=self.bilstm.output_dim,  # 512 (256*2)
            num_heads=4,
            dropout=dropout
        )

        # 4. Classification Head
        self.classifier = ClassificationHead(
            input_dim=self.bilstm.output_dim,
            hidden_dims=[512, 256],
            num_classes=num_classes,
            dropout=dropout
        )

    def forward(self, esm_embeddings, prost_embeddings, attention_mask=None):
        """
        Args:
            esm_embeddings: (batch, seq_len, 1280)
            prost_embeddings: (batch, seq_len, 1024)
            attention_mask: (batch, seq_len)

        Returns:
            logits: (batch, num_classes)
            attention_weights: (batch, num_heads, seq_len)
        """
        # 1. Cross-attention bidirectionnelle
        fused = self.cross_attention(esm_embeddings, prost_embeddings, attention_mask)

        # 2. BiLSTM pour dependances sequentielles
        lstm_out = self.bilstm(fused, attention_mask)

        # 3. Multi-head attention pooling
        pooled, attn_weights = self.attention_pooling(lstm_out, attention_mask)

        # 4. Classification
        logits = self.classifier(pooled)

        return logits, attn_weights


class SimpleCrossAttentionNetwork(nn.Module):
    """
    Version simplifiee avec juste Cross-Attention (sans BiLSTM)
    Pour comparaison et tests rapides
    """

    def __init__(
        self,
        esm_dim: int = 1280,
        prost_dim: int = 1024,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_classes: int = 6,
        dropout: float = 0.3
    ):
        super().__init__()

        self.cross_attention = BidirectionalCrossAttention(
            esm_dim=esm_dim,
            prost_dim=prost_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        self.attention_pooling = MultiHeadAttentionPooling(
            embedding_dim=hidden_dim,
            num_heads=4,
            dropout=dropout
        )

        self.classifier = ClassificationHead(
            input_dim=hidden_dim,
            hidden_dims=[512, 256],
            num_classes=num_classes,
            dropout=dropout
        )

    def forward(self, esm_embeddings, prost_embeddings, attention_mask=None):
        fused = self.cross_attention(esm_embeddings, prost_embeddings, attention_mask)
        pooled, attn_weights = self.attention_pooling(fused, attention_mask)
        logits = self.classifier(pooled)
        return logits, attn_weights


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the improved models
    print("="*60)
    print("Testing Improved Fusion Networks")
    print("="*60)

    batch_size = 4
    seq_len = 100

    # Create dummy data (ESM-2 650M dimensions)
    esm_emb = torch.randn(batch_size, seq_len, 1280)
    prost_emb = torch.randn(batch_size, seq_len, 1024)
    mask = torch.ones(batch_size, seq_len)
    mask[:, 80:] = 0  # Simulate padding

    # Test ImprovedFusionNetwork (full)
    print("\n1. Testing ImprovedFusionNetwork (Cross-Attn + BiLSTM)...")
    model_full = ImprovedFusionNetwork(
        esm_dim=1280,
        prost_dim=1024,
        hidden_dim=512,
        dropout=0.2
    )
    logits, attn = model_full(esm_emb, prost_emb, mask)
    print(f"   Logits shape: {logits.shape}")
    print(f"   Attention shape: {attn.shape}")
    print(f"   Parameters: {count_parameters(model_full):,}")

    # Test SimpleCrossAttentionNetwork
    print("\n2. Testing SimpleCrossAttentionNetwork (Cross-Attn only)...")
    model_simple = SimpleCrossAttentionNetwork(
        esm_dim=1280,
        prost_dim=1024,
        hidden_dim=512,
        dropout=0.2
    )
    logits, attn = model_simple(esm_emb, prost_emb, mask)
    print(f"   Logits shape: {logits.shape}")
    print(f"   Attention shape: {attn.shape}")
    print(f"   Parameters: {count_parameters(model_simple):,}")

    print("\n" + "="*60)
    print("All tests passed!")
    print("="*60)
