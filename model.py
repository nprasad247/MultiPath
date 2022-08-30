from layers import MultiPathLayer, MultiPathMLP
import pytorch_lightning as pl
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR as CosineAnnealing
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric import nn as gnn
from torchmetrics.functional import confusion_matrix, auroc, f1_score as f1



class MultiPathModel(pl.LightningModule):


    def __init__(self, **config):
    
        super().__init__()
        """
        Define model layers.
        """
        in_channels = config['in_channels']
        hidden_dim = config['hidden_dim']
        num_hidden_layers = config['num_hidden_layers']
        out_channels = config['out_channels']
        activation = config['activation']
        self.init_proj = nn.Embedding(53, hidden_dim)
        self.hidden_layers = [(MultiPathLayer(hidden_dim, hidden_dim, batch_norm=True, activation=activation), 'x, edge_index, bond_types, coords -> x') for _ in range(num_hidden_layers - 1)]
        self.hidden = gnn.Sequential('x, edge_index, bond_types, coords', self.hidden_layers)
        self.final_conv = MultiPathLayer(hidden_dim, hidden_dim, batch_norm=True, final=True, activation=activation)
    

        self.out = MultiPathMLP(in_channels=hidden_dim, out_channels=out_channels, activation=nn.ReLU(), num_hidden_layers=3)
        
        self.loss = nn.CrossEntropyLoss()

        
    def forward(self, data):
        """
        Define the forward pass for the model.
        """
        x, edge_index, bond_types, coords = data.x, data.edge_index, data.bond_types, data.coords

        x = self.init_proj(x).squeeze(1)

        x = self.hidden(x, edge_index, bond_types, coords)
        x = self.final_conv(x, edge_index, bond_types, coords)
        x = gnn.global_add_pool(x, data.batch)
        return self.out(x)


    def configure_optimizers(self):
        """
        Set the optimizer for the model.
        """

        optimizer = torch.optim.AdamW(self.parameters(), lr=.001, weight_decay = 10e-9)

        lr_scheduler = CosineAnnealing(optimizer, warmup_epochs=100, max_epochs=1000)
        scheduler = {
            'scheduler': lr_scheduler,
            'reduce_on_plateau': True,
            'monitor': 'val_loss'
        }

        return [optimizer], [scheduler]

    
    def classification_metrics(self, y_hat, y):
        y_probs = F.softmax(y_hat, dim=1)
        conf_matrix = confusion_matrix(y_probs, y, num_classes=2)
        total = len(y)
        total_positives = conf_matrix[1].sum()
        false_negatives = conf_matrix[1, 0]
        false_positives = conf_matrix[0, 1]
        total_negatives = total - total_positives
        
        results = {
            "correct" : torch.diag(conf_matrix).sum(),
            "total" : total,
            "F1" : f1(y_probs, y),
            "false_negative" : false_negatives,
            "total_positive" : total_positives,
            "false_positive" : false_positives,
            "total_negative" : total_negatives,
            "y_probs" : y_probs
        }
        return results

    
    def model_step(self, batch, batch_idx, mode):
        """
        Function to handle training and validation steps.
        """
        y = batch.y
        y_hat = self(batch)
        loss = self.loss(y_hat, y)
        logs = {f'{mode}_loss' : loss}

        batch_dict = {
            "loss" : loss,
            "log" : logs,
            "y" : y,
            "y_hat" : y_hat.argmax(dim=1),
            **self.classification_metrics(y_hat, y)
        }

        return batch_dict


    def training_step(self, train_batch, batch_idx):
        """
        A single training step for the model.
        """
        return self.model_step(train_batch, batch_idx, 'train')


    def validation_step(self, val_batch, batch_idx):
        """
        A single validation step for the model.
        """
        return self.model_step(val_batch, batch_idx, 'val')

