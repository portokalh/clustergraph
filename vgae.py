from typing import Optional, Callable, List

from deeplay.applications import Application
from deeplay.external import Optimizer, Adam

from deeplay import (
    DeeplayModule,
    Layer,
)

import torch
import torch.nn as nn


class VariationalGraphAutoEncoder(Application):
    """Variational Auto Encoder for Graphs (GAUDI-style)

    Parameters
    ----------
    encoder : nn.Module
        Graph encoder that takes a dict (x, edge_attr, edge_index, batch, ...)
        and returns a dict with updated 'x' and pooling losses (L_cut/L_ortho).
    decoder : nn.Module
        Graph decoder that reconstructs node and edge features from 'x'.
    hidden_features : int
        Dimensionality of the encoder output features ('x') that feed the latent layers.
    latent_dim : int
        Latent dimension size.
    alpha : float
        Weight for node feature reconstruction loss.
    beta : float
        Weight for KL divergence term.
    gamma : float
        Weight for edge feature reconstruction loss.
    delta : float
        Weight for pooling (MinCut / orthogonality) losses.
    reconstruction_loss : Callable
        Loss for reconstruction (default L1).
    pool_loss_terms : list[str]
        List of pooling loss keys, e.g. ['L_cut', 'L_ortho'].
    optimizer : Optimizer
        Deeplay optimizer wrapper.
    """

    encoder: torch.nn.Module
    decoder: torch.nn.Module
    hidden_features: int
    latent_dim: int
    alpha: float
    beta: float
    gamma: float
    delta: float
    reconstruction_loss: torch.nn.Module
    pool_loss_terms: List[str]
    optimizer: Optimizer

    def __init__(
        self,
        encoder: Optional[nn.Module] = None,
        decoder: Optional[nn.Module] = None,
        hidden_features: int = 96,
        latent_dim: int = 8,
        alpha: float = 1.0,
        beta: float = 1e-7,
        gamma: float = 1.0,
        delta: float = 1.0,
        reconstruction_loss: Optional[Callable] = None,
        pool_loss_terms: Optional[List[str]] = None,
        optimizer: Optional[Optimizer] = None,
        **kwargs,
    ):
        self.encoder = encoder
        self.decoder = decoder
        self.hidden_features = hidden_features
        self.latent_dim = latent_dim

        # Use L1 by default for reconstruction
        self.reconstruction_loss = reconstruction_loss or nn.L1Loss()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

        # Default pooling losses (MinCut + orthogonality)
        self.pool_loss_terms = pool_loss_terms or ["L_cut", "L_ortho"]

        # ---- Latent mapping layers (Deeplay Layer wrappers) ----
        # encoder output 'x' (size = hidden_features) -> mu/log_var (latent_dim)
        self.fc_mu = Layer(nn.Linear, hidden_features, latent_dim)
        self.fc_mu.set_input_map("x")
        self.fc_mu.set_output_map("mu")

        self.fc_var = Layer(nn.Linear, hidden_features, latent_dim)
        self.fc_var.set_input_map("x")
        self.fc_var.set_output_map("log_var")

        # decoder input: 'z' (latent_dim) -> 'x' (hidden_features), then decoder reconstructs
        self.fc_dec = Layer(nn.Linear, latent_dim, hidden_features)
        self.fc_dec.set_input_map("z")
        self.fc_dec.set_output_map("x")

        # Initialize Application (Lightning plumbing, logging, etc.)
        super().__init__(**kwargs)

        # ---- Reparameterization module (mu, log_var -> z) ----
        class Reparameterize(DeeplayModule):
            def forward(self, mu, log_var):
                # Clamp for numerical stability
                mu_safe = torch.clamp(mu, -10.0, 10.0)
                log_var_safe = torch.clamp(log_var, -10.0, 10.0)

                std = torch.exp(0.5 * log_var_safe)
                eps = torch.randn_like(std)
                return eps * std + mu_safe

        self.reparameterize = Reparameterize()
        self.reparameterize.set_input_map("mu", "log_var")
        self.reparameterize.set_output_map("z")

        # ---- Optimizer ----
        self.optimizer = optimizer or Adam(lr=1e-3)

        @self.optimizer.params
        def params(self):
            return self.parameters()

    # ------------------------------------------------------------------
    # Encode / decode
    # ------------------------------------------------------------------
    def encode(self, x):
        """
        x: dict with keys like 'x', 'edge_attr', 'edge_index', 'batch', ...
        Returns dict with added 'mu' and 'log_var'.
        """
        x = self.encoder(x)   # GAUDI GraphEncoder; outputs dict with 'x' and pool terms
        x = self.fc_mu(x)     # adds 'mu'
        x = self.fc_var(x)    # adds 'log_var'
        return x

    def decode(self, x):
        """
        x: dict with 'z' present, plus graph structure.
        Returns dict with reconstructed 'x' and 'edge_attr' (through decoder).
        """
        x = self.fc_dec(x)    # uses 'z' -> 'x'
        x = self.decoder(x)   # reconstructs node/edge features, pool terms remain
        return x

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        # Let Application handle moving to device, etc.
        x, y = self.train_preprocess(batch)
        node_features, edge_features = y  # targets

        # Forward VAE
        x = self(x)

        node_features_hat = x["x"]
        edge_features_hat = x["edge_attr"]
        mu = x["mu"]
        log_var = x["log_var"]

        # Pooling loss (MinCut & orthogonality) – sum across all clusters/blocks
        pool_loss = 0.0
        loss_details = {}
        for term in self.pool_loss_terms:
            term_loss = sum(
                value for key, value in x.items() if key.startswith(term)
            )
            loss_details[f"{term} loss"] = term_loss
            pool_loss = pool_loss + term_loss

        # Reconstruction + KL (stable)
        rec_loss_nodes, rec_loss_edges, KLD = self.compute_loss(
            node_features_hat,
            node_features,
            edge_features_hat,
            edge_features,
            mu,
            log_var,
        )

        total_loss = (
            self.alpha * rec_loss_nodes
            + self.gamma * rec_loss_edges
            + self.beta * KLD
            + self.delta * pool_loss
        )

        loss_details.update(
            {
                "rec_loss_nodes": rec_loss_nodes,
                "rec_loss_edges": rec_loss_edges,
                "KL": KLD,
                "total_loss": total_loss,
            }
        )

        # Log everything with Lightning-style names
        for name, v in loss_details.items():
            self.log(
                f"train_{name}",
                v,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

        return total_loss

    # ------------------------------------------------------------------
    # Stable loss
    # ------------------------------------------------------------------
    def compute_loss(self, n_hat, n, e_hat, e, mu, log_var):
        """
        Stable VAE loss:
        - NaN/Inf-protected reconstructions
        - Clamped mu/log_var
        - Mean KL (not sum) to avoid huge magnitudes
        """

        # --- NaN/Inf protection on recon targets and predictions ---
        n_hat = torch.nan_to_num(n_hat, nan=0.0, posinf=1e6, neginf=-1e6)
        e_hat = torch.nan_to_num(e_hat, nan=0.0, posinf=1e6, neginf=-1e6)
        n = torch.nan_to_num(n, nan=0.0, posinf=1e6, neginf=-1e6)
        e = torch.nan_to_num(e, nan=0.0, posinf=1e6, neginf=-1e6)

        rec_loss_nodes = self.reconstruction_loss(n_hat, n)
        rec_loss_edges = self.reconstruction_loss(e_hat, e)

        # --- Stable KL divergence ---
        mu_safe = torch.clamp(mu, -10.0, 10.0)
        log_var_safe = torch.clamp(log_var, -10.0, 10.0)

        # Standard VAE KL, but elementwise -> mean to keep scale under control
        kld_element = -0.5 * (
            1.0 + log_var_safe - mu_safe.pow(2.0) - log_var_safe.exp()
        )
        KLD = kld_element.mean()

        # Final NaN/Inf guard
        if torch.isnan(KLD) or torch.isinf(KLD):
            KLD = torch.zeros((), device=mu.device)

        return rec_loss_nodes, rec_loss_edges, KLD

    # ------------------------------------------------------------------
    # Forward = encode → reparameterize → decode
    # ------------------------------------------------------------------
    def forward(self, x):
        # encode
        x = self.encode(x)
    
        # ensure everything is on same device
        device = next(self.parameters()).device
        for k, v in x.items():
            if torch.is_tensor(v):
                x[k] = v.to(device)
    
        # reparameterize
        x = self.reparameterize(x)
    
        # decode
        x = self.decode(x)
    
        return x
