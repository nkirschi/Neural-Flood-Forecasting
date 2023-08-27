from torch import Tensor
from torch.nn import Module
from torch.nn.utils.parametrize import register_parametrization
from torch.nn.functional import relu
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.utils import spmm


class GRAFFConv(MessagePassing):

    def __init__(self, channels, step_size=1, add_self_loops=True, normalize=True, **kwargs):
        kwargs.setdefault("aggr", "add")
        super().__init__(**kwargs)

        self.channels = channels
        self.step_size = step_size
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self.internal_mixer = SymmetricLinear(channels)
        self.external_mixer = SymmetricLinear(channels)
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.internal_mixer.reset_parameters()
        self.external_mixer.reset_parameters()

    def forward(self, x, edge_index, edge_weight=None):
        edge_index, edge_weight = gcn_norm(edge_index, edge_weight, x.size(self.node_dim), False,
                                           self.add_self_loops, self.flow, dtype=x.dtype)
        internal_repr = self.propagate(edge_index, x=self.internal_mixer(x), edge_weight=edge_weight)
        external_repr = self.external_mixer(x)
        return x + self.step_size * relu(internal_repr - external_repr)

    def message(self, x_j, edge_weight) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t, x) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.channels})"


class SymmetricLinear(Linear):
    class Symmetrizer(Module):
        def forward(self, x):
            return (x + x.transpose(-1, -2)) / 2

    def __init__(self, channels):
        super().__init__(channels, channels, bias=False, weight_initializer="glorot")
        register_parametrization(self, "weight", self.Symmetrizer())
