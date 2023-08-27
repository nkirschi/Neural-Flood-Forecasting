from typing import Any, Callable, Dict, Optional, Union

from torch.nn import Module, ModuleList
from torch_geometric.nn import MLP
from torch_geometric.typing import OptTensor

from graff_conv import GRAFFConv


class GRAFFNN(Module):
    def __init__(self,
                 num_layers: int,
                 in_channels: int,
                 out_channels: int,
                 hidden_channels: int,
                 shared_weights: bool = True,
                 step_size: int = 1,
                 act: Union[str, Callable, None] = 'relu',
                 act_kwargs: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.encoder = MLP([in_channels, hidden_channels])
        self.decoder = MLP([hidden_channels, out_channels])
        if shared_weights:
            self.graff_convs = ModuleList(num_layers * [GRAFFConv(channels=hidden_channels, step_size=step_size,
                                                                  act=act, act_kwargs=act_kwargs)])
        else:
            self.graff_convs = ModuleList([GRAFFConv(channels=hidden_channels, step_size=step_size,
                                                     act=act, act_kwargs=act_kwargs)
                                           for _ in range(num_layers)])

    def forward(self, x, edge_index, edge_weight: OptTensor = None):
        x_0 = self.encoder(x)
        x = x_0
        for graff_conv in self.graff_convs:
            x = graff_conv(x, x_0, edge_index, edge_weight)
        return self.decoder(x)
