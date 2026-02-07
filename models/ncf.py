import torch
import torch.nn as nn

class NCF(nn.Module):
    """
    Neural Collaborative Filtering (NCF) Model.
    Combines Generalized Matrix Factorization (GMF) and Multi-Layer Perceptron (MLP).
    """
    def __init__(self, num_users, num_items, embedding_dim=32, layers=[64, 32, 16], dropout=0.2):
        super(NCF, self).__init__()
        
        self.num_users = num_users
        self.num_items = num_items
        
        # --- GMF Part ---
        self.embedding_user_gmf = nn.Embedding(num_users, embedding_dim)
        self.embedding_item_gmf = nn.Embedding(num_items, embedding_dim)
        
        # --- MLP Part ---
        self.embedding_user_mlp = nn.Embedding(num_users, layers[0] // 2)
        self.embedding_item_mlp = nn.Embedding(num_items, layers[0] // 2)
        
        # MLP Layers
        self.mlp_layers = nn.ModuleList()
        # Input to first layer is concatenation of user + item embeddings
        input_size = layers[0] 
        
        for layer_size in layers[1:]:
            self.mlp_layers.append(nn.Linear(input_size, layer_size))
            self.mlp_layers.append(nn.ReLU())
            self.mlp_layers.append(nn.Dropout(dropout))
            input_size = layer_size
            
        # --- Output Layer ---
        # Concatenate GMF (embedding_dim) and MLP (last layer size)
        final_input_size = embedding_dim + layers[-1]
        self.output_layer = nn.Linear(final_input_size, 1) # Regression output (rating)
        
        # Weight Initialization
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights with He (Kaiming) initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.01)

    def forward(self, user_indices, item_indices):
        # GMF Forward
        user_gmf = self.embedding_user_gmf(user_indices)
        item_gmf = self.embedding_item_gmf(item_indices)
        vector_gmf = torch.mul(user_gmf, item_gmf) # Element-wise product
        
        # MLP Forward
        user_mlp = self.embedding_user_mlp(user_indices)
        item_mlp = self.embedding_item_mlp(item_indices)
        vector_mlp = torch.cat([user_mlp, item_mlp], dim=-1) # Concatenation
        
        for layer in self.mlp_layers:
            vector_mlp = layer(vector_mlp)
            
        # NeuMF (Concatenate GMF and MLP)
        vector_final = torch.cat([vector_gmf, vector_mlp], dim=-1)
        output = self.output_layer(vector_final)
        
        return output.squeeze()
