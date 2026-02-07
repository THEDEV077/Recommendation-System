import torch
import torch.nn as nn

class AutoRec(nn.Module):
    """
    AutoRec: Autoencoder for Collaborative Filtering.
    Item-based AutoRec (I-AutoRec) takes item vectors (ratings from all users) 
    and reconstructs them to predict missing ratings.
    """
    def __init__(self, num_inputs, hidden_dim=500, dropout=0.05):
        super(AutoRec, self).__init__()
        
        self.encoder = nn.Linear(num_inputs, hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.decoder = nn.Linear(hidden_dim, num_inputs)
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
        
    def _init_weights(self):
        # Xavier initialization
        nn.init.xavier_normal_(self.encoder.weight)
        nn.init.xavier_normal_(self.decoder.weight)
        nn.init.constant_(self.encoder.bias, 0)
        nn.init.constant_(self.decoder.bias, 0)
        
    def forward(self, x):
        # x is (batch_size, num_inputs)
        # Encoder
        latent = self.sigmoid(self.encoder(x))
        latent = self.dropout(latent)
        
        # Decoder
        # AutoRec output is usually identity activation (linear)
        reconstruction = self.decoder(latent) 
        
        return reconstruction
