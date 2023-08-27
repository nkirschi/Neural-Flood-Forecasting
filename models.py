import torch

from abc import ABC, abstractmethod
from graffconv import GRAFFConv
from torch.nn import Module, ModuleList
from torch.nn.functional import mse_loss, relu
from torch_geometric.nn import GCNConv, GCN2Conv, Linear
from torch_geometric.utils import add_self_loops


class BaseModel(Module, ABC):
    def __init__(self, in_channels, hidden_channels, num_hidden, param_sharing, layerfun, edge_orientation, edge_weights):
        super().__init__()
        self.encoder = Linear(in_channels, hidden_channels, weight_initializer="glorot")
        self.decoder = Linear(hidden_channels, 1, weight_initializer="glorot")
        if param_sharing:
            self.layers = ModuleList(num_hidden * [layerfun()])
        else:
            self.layers = ModuleList([layerfun() for _ in range(num_hidden)])
        self.edge_weights = edge_weights
        self.edge_orientation = edge_orientation
        self.loop_fill_value = 1.0 if (self.edge_weights == 0).all() else "mean"

    def forward(self, x, edge_index, y=None, evo_tracking=False):
        if self.edge_weights is not None:
            num_graphs = edge_index.size(1) // len(self.edge_weights)
            edge_weights = self.edge_weights.repeat(num_graphs).to(x.device)
            edge_weights = edge_weights.clamp(min=0)  # relevant when edge weights are learned
        else:
            edge_weights = torch.zeros(edge_index.size(1)).to(x.device)

        if self.edge_orientation is not None:
            if self.edge_orientation == "upstream":
                edge_index = edge_index[[1, 0]].to(x.device)
            elif self.edge_orientation == "bidirectional":
                edge_index = torch.cat([edge_index, edge_index[[1, 0]]], dim=1).to(x.device)
                edge_weights = edge_weights.repeat(2).to(x.device)
            elif self.edge_orientation != "downstream":
                raise ValueError("unknown edge direction", self.edge_orientation)
        edge_index, edge_weights = add_self_loops(edge_index, edge_weights, fill_value=self.loop_fill_value,
                                                  num_nodes=x.size(0))  # must be specified for IPUs

        x_0 = self.encoder(x)
        evolution = [x_0.detach()] if evo_tracking else None

        x = x_0
        for layer in self.layers:
            x = self.apply_layer(layer, x, x_0, edge_index, edge_weights)
            if evo_tracking:
                evolution.append(x.detach())
        x = self.decoder(x)

        if evo_tracking:
            return x, evolution
        if y is not None:  # IPU paradigm demands that models return the loss as 2nd output
            return x, mse_loss(x, y)
        return x

    @abstractmethod
    def apply_layer(self, layer, x, x_0, edge_index, edge_weights):
        pass


class MLP(BaseModel):
    def __init__(self, in_channels, hidden_channels, num_hidden, param_sharing):
        layer_gen = lambda: Linear(hidden_channels, hidden_channels, weight_initializer="glorot")
        super().__init__(in_channels, hidden_channels, num_hidden, param_sharing, layer_gen, None, None)

    def apply_layer(self, layer, x, x_0, edge_index, edge_weights):
        return relu(layer(x))


class GCN(BaseModel):
    def __init__(self, in_channels, hidden_channels, num_hidden, param_sharing, edge_orientation, edge_weights):
        layer_gen = lambda: GCNConv(hidden_channels, hidden_channels, add_self_loops=False)
        super().__init__(in_channels, hidden_channels, num_hidden, param_sharing, layer_gen, edge_orientation, edge_weights)

    def apply_layer(self, layer, x, x_0, edge_index, edge_weights):
        return relu(layer(x, edge_index, edge_weights))


class ResGCN(GCN):
    def __init__(self, in_channels, hidden_channels, num_hidden, param_sharing, edge_orientation, edge_weights):
        super().__init__(in_channels, hidden_channels, num_hidden, param_sharing, edge_orientation, edge_weights)

    def apply_layer(self, layer, x, x_0, edge_index, edge_weights):
        return x + super().apply_layer(layer, x, x_0, edge_index, edge_weights)


class GCNII(BaseModel):
    def __init__(self, in_channels, hidden_channels, num_hidden, param_sharing, edge_orientation, edge_weights):
        layer_gen = lambda: GCN2Conv(hidden_channels, alpha=0.5, add_self_loops=False)
        super().__init__(in_channels, hidden_channels, num_hidden, param_sharing, layer_gen, edge_orientation, edge_weights)

    def apply_layer(self, layer, x, x_0, edge_index, edge_weights):
        return relu(layer(x, x_0, edge_index, edge_weights))


class GRAFFNN(BaseModel):
    def __init__(self, in_channels, hidden_channels, num_hidden, param_sharing, step_size, edge_orientation, edge_weights):
        layer_gen = lambda: GRAFFConv(channels=hidden_channels, step_size=step_size, add_self_loops=False)
        super().__init__(in_channels, hidden_channels, num_hidden, param_sharing, layer_gen, edge_orientation, edge_weights)

    def apply_layer(self, layer, x, x_0, edge_index, edge_weights):
        return layer(x, edge_index, edge_weights)  # GRAFFConv already includes non-linearity
