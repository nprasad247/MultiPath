import torch
from torch import nn
from torch_geometric import nn as gnn



class MultiPathLayer(nn.Module):


    def __init__(self, in_channels, out_channels, activation='mish', final=False, batch_norm=True, agg_mode='att', features=["node", "edge", "struct"]):
        
        super(MultiPathLayer, self).__init__()
        self.final = final
        self.features = features
        
        if type(activation) != str:
            self.activation = activation
        
        else:
            
            if activation == 'sigmoid':
                self.activation = nn.Sigmoid()
            if activation == 'relu':
                self.activation = nn.ReLU()
            if activation == 'silu':
                self.activation = nn.SiLU()
            if activation == 'mish':
                self.activation = nn.Mish()
            
        if "node" in features:
            self.node_features = NodeFeaturePath(in_channels, out_channels, self.activation)
            
        if "edge" in features:
            self.edge_features = EdgeFeaturePath(in_channels, out_channels, self.activation)
            
        if "struct" in features:
            self.struct_features = StructuralFeaturePath(in_channels, out_channels, self.activation)
            
        self.feat_aggregator = FeatureAggregation(out_channels, out_channels, self.activation, mode=agg_mode, num_features=len(features))
        self.batch_norm = gnn.BatchNorm(out_channels) if batch_norm else nn.Identity()
        
        
    def forward(self, x, edge_index, bond_types, coords):
        
        features = []
        if "node" in self.features:
            features.append(self.node_features(x, edge_index))
            
        if "edge" in self.features:
            features.append(self.edge_features(x, edge_index, bond_types))
            
        if "struct" in self.features:
            features.append(self.struct_features(x, edge_index, coords))
            
        h_agg = self.feat_aggregator(x, features)
        h_agg = self.batch_norm(h_agg)
        return h_agg if self.final else self.activation(h_agg)
        
        
        
        
class NodeFeaturePath(gnn.MessagePassing):
    
    
    def __init__(self, in_channels, out_channels, activation):
    
        super().__init__(aggr='add')
        self.neighbor_lin = nn.Linear(2 * in_channels, out_channels)
        self.out_lin = nn.Linear(out_channels, out_channels)
        self.activation = activation
        
        
    def forward(self, x, edge_index):

        out = self.propagate(edge_index, x=x)
        return self.activation(self.out_lin(out))
        
        
    def message(self, x_i, x_j):
        
        pair_ij = torch.cat([x_i, x_j], dim=1)
        pair_ij = self.activation(self.neighbor_lin(pair_ij))
        return pair_ij



class EdgeFeaturePath(gnn.MessagePassing):

    
    def __init__(self, in_channels, out_channels, activation):
    
        super().__init__(aggr='add')
        self.activation = activation
        self.edge_conv = nn.Conv2d(4, 1, 1)
        self.edge_lin = nn.Linear(2 * in_channels, out_channels)
        self.out_lin = nn.Linear(out_channels, out_channels)
        
        
    def edge_feature_norm(self, x, edge_index, bond_types):


        adj = torch.zeros(4, len(x), len(x), device=x.device)

        row, col = edge_index
        adj[bond_types, row, col] = 1
        E = self.edge_conv(adj).squeeze(0)

        return E[row, col].view(-1, 1)
    

    def forward(self, x, edge_index, bond_types):
       
        norm = self.edge_feature_norm(x, edge_index, bond_types)
        P = self.propagate(edge_index, x=x, norm=norm)
        return self.activation(self.out_lin(P))
        
        
    def message(self, x_i, x_j, norm):
    
        pair_ij = torch.cat([x_i, x_j], dim=1)
        pair_ij = norm * pair_ij
        return self.activation(self.edge_lin(pair_ij))
    
    
    
class StructuralFeaturePath(gnn.MessagePassing):


    def __init__(self, in_channels, out_channels, activation):
        
        super().__init__(aggr='add')
        self.coord_lin = nn.Linear(2 * in_channels, out_channels)
        self.activation = activation
        self.pair_lin = nn.Linear(out_channels, out_channels)
        self.out_lin = nn.Linear(2 * out_channels, out_channels)
        
    
    def forward(self, x, edge_index, coords):
       
        row, col= edge_index
        norm = ((coords[row] - coords[col]) ** 2).sum(dim=1).view(-1, 1)
        Q = self.propagate(edge_index, x=x, norm=norm)
        Q = self.pair_lin(Q)
        Q = self.activation(Q)
        x_out = torch.cat([x, Q], dim=1)
        return self.activation(self.out_lin(x_out))
        
    
    def message(self, x_i, x_j, norm):
        
        norm_x_cat = norm * torch.cat([x_i, x_j], dim=1)
        pair_ij = self.coord_lin(norm_x_cat)
        return pair_ij



class FeatureAggregation(nn.Module):


    def __init__(self, in_channels, out_channels, activation, mode='att', num_features=3):
    
        super(FeatureAggregation, self).__init__()
        self.activation = activation
        
        if mode == 'concat':
            self.concat_lin = nn.Linear(num_features * in_channels, out_channels, bias=False)
            return
            
        if mode == 'att':
        
            self.init_lin = nn.Linear(in_channels, in_channels, bias=False)
            self.feat_lin = nn.Parameter(torch.randn(num_features, in_channels, in_channels))
            torch.nn.init.xavier_uniform_(self.feat_lin)
            self.att = nn.Linear(2 * in_channels, out_channels, bias=False)
            
        self.agg_lin = nn.Linear(in_channels, out_channels, bias=False)
        self.mode = mode
        
        
    def forward(self, x, features):
    
        
        if self.mode == 'att':
            
            lin_init = self.init_lin(x).repeat(len(features), 1, 1)
            feat_stack = torch.stack(features, dim=0)
           
            lin_feats = torch.bmm(feat_stack, self.feat_lin)
            
            lin_feats = torch.cat([self.activation(lin_init), self.activation(lin_feats)], dim=2)
            e = self.att(self.activation(lin_feats))
            a = torch.softmax(e, dim=0)

            att = torch.einsum("fnd, fnd -> nd", a, feat_stack)
            return self.agg_lin(att)
            
        if self.mode == 'sum':
            out = self.agg_lin(sum(features))
            return self.activation(out)
                
        if self.mode == 'max':
            
            out = features[0]
            for feature in features:
                out = torch.maximum(out, feature)
            out = self.agg_lin(out)
            return self.activation(out)
            
        if self.mode == 'concat':
            out = self.concat_lin(torch.cat(features, dim=1))
            return self.activation(out)
            
            
        raise ValueError(f"Error: {self.mode} not in ['att', 'sum', 'max', 'concat']")
            
            
            
class MultiPathMLP(nn.Module):

  
    def __init__(self, **config):
        
        super(MultiPathMLP, self).__init__()
        in_channels = config['in_channels']
        out_channels = config['out_channels']
        activation = config['activation']
        num_hidden_layers = config['num_hidden_layers']
        self.layer1 = nn.Linear(in_channels, out_channels, bias=False)
        self.norm = nn.BatchNorm1d(out_channels)
        self.activation = activation
        
        hidden_layers = []
        
        for _ in range(num_hidden_layers - 1):
            hidden_layers.extend([nn.Linear(out_channels, out_channels, bias=False), 
                                  nn.BatchNorm1d(out_channels), 
                                  self.activation])
            
        self.hidden_layers = nn.Sequential(*hidden_layers)
        self.out = nn.Linear(out_channels, out_channels, bias=False)
        self.final_norm = nn.BatchNorm1d(out_channels)
        
        
    def forward(self, x):
    
        x = self.layer1(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.hidden_layers(x)
        x = self.out(x)
        return self.final_norm(x)
