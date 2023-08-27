import copy
import numpy as np
import os
import poptorch
import random
import torch
import torch.nn as nn

from dataset import LamaHDataset
from datetime import datetime
from models import MLP, GCN, ResGCN, GCNII, GRAFFNN
from torch.nn.functional import mse_loss
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
from torch_geometric.utils import get_laplacian, to_undirected, to_torch_coo_tensor

from torchinfo import summary
from tqdm import tqdm


def ensure_reproducibility(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_edge_weights(adjacency_type, edge_attr):
    if adjacency_type == "isolated":
        return torch.zeros(edge_attr.size(0))
    elif adjacency_type == "binary":
        return torch.ones(edge_attr.size(0))
    elif adjacency_type == "stream_length":
        return edge_attr[:, 0]
    elif adjacency_type == "elevation_difference":
        return edge_attr[:, 1]
    elif adjacency_type == "average_slope":
        return edge_attr[:, 2]
    elif adjacency_type == "learned":
        return nn.Parameter(torch.nn.init.uniform_(torch.empty(edge_attr.size(0)), 0.9, 1.1))
    else:
        raise ValueError("invalid adjacency type", adjacency_type)


def construct_model(hparams, dataset):
    edge_weights = get_edge_weights(hparams["model"]["adjacency_type"], dataset.edge_attr)
    model_arch = hparams["model"]["architecture"]
    if model_arch == "MLP":
        return MLP(in_channels=hparams["data"]["window_size"],
                   hidden_channels=hparams["model"]["hidden_channels"],
                   num_hidden=hparams["model"]["num_layers"],
                   param_sharing=hparams["model"]["param_sharing"])
    elif model_arch == "GCN":
        return GCN(in_channels=hparams["data"]["window_size"],
                   hidden_channels=hparams["model"]["hidden_channels"],
                   num_hidden=hparams["model"]["num_layers"],
                   param_sharing=hparams["model"]["param_sharing"],
                   edge_orientation=hparams["model"]["edge_orientation"],
                   edge_weights=edge_weights
                   )
    elif model_arch == "ResGCN":
        return ResGCN(in_channels=hparams["data"]["window_size"],
                      hidden_channels=hparams["model"]["hidden_channels"],
                      num_hidden=hparams["model"]["num_layers"],
                      param_sharing=hparams["model"]["param_sharing"],
                      edge_orientation=hparams["model"]["edge_orientation"],
                      edge_weights=edge_weights)
    elif model_arch == "GCNII":
        return GCNII(in_channels=hparams["data"]["window_size"],
                     hidden_channels=hparams["model"]["hidden_channels"],
                     num_hidden=hparams["model"]["num_layers"],
                     param_sharing=hparams["model"]["param_sharing"],
                     edge_orientation=hparams["model"]["edge_orientation"],
                     edge_weights=edge_weights)
    elif model_arch == "GRAFFNN":
        return GRAFFNN(in_channels=hparams["data"]["window_size"],
                       hidden_channels=hparams["model"]["hidden_channels"],
                       num_hidden=hparams["model"]["num_layers"],
                       param_sharing=hparams["model"]["param_sharing"],
                       step_size=hparams["model"]["graff_step_size"],
                       edge_orientation=hparams["model"]["edge_orientation"],
                       edge_weights=edge_weights
                       )
    raise ValueError("unknown model architecture", model_arch)


def load_dataset(hparams, split):
    if split == "train":
        years = range(2000, 2016)
    elif split == "test":
        years = [2016, 2017]
    else:
        raise ValueError("unknown split", split)
    return LamaHDataset("LamaH-CE",
                        years=years,
                        window_size=hparams["data"]["window_size"],
                        stride_length=hparams["data"]["stride_length"],
                        lead_time=hparams["data"]["lead_time"],
                        normalized=hparams["data"]["normalized"])


def load_model_and_dataset(chkpt):
    best_epoch = torch.tensor(chkpt["history"]["val_loss"]).argmin()
    model_params = chkpt["history"]["model_params"][best_epoch]
    dataset = load_dataset(chkpt["hparams"], "test")
    model = construct_model(chkpt["hparams"], dataset)
    model.load_state_dict(model_params, strict=False)
    return model, dataset


def train_step(model, train_loader, optimizer, device):
    model.train()
    train_loss = 0
    for batch in tqdm(train_loader, desc="Training"):
        if device != "ipu":
            batch = batch.to(device)
            optimizer.zero_grad()
        out, loss = model(batch.x, batch.edge_index, batch.y)
        if device != "ipu":
            loss.backward()
            optimizer.step()
        train_loss += loss.item() * batch.num_graphs / len(train_loader.dataset)
    return train_loss


def val_step(model, val_loader, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            if device != "ipu":
                batch = batch.to(device)
            out, loss = model(batch.x, batch.edge_index, batch.y)
            val_loss += loss.item() * batch.num_graphs / len(val_loader.dataset)
    return val_loss


def train(model, dataset, hparams, save_dir="runs/", on_ipu=False):
    print(summary(model, depth=2))

    holdout_size = hparams["training"]["holdout_size"]
    train_dataset, val_dataset = random_split(dataset, [1 - holdout_size, holdout_size])
    train_loader = DataLoader(train_dataset, batch_size=hparams["training"]["batch_size"], shuffle=True,
                              drop_last=on_ipu)
    val_loader = DataLoader(val_dataset, batch_size=hparams["training"]["batch_size"], shuffle=False, drop_last=on_ipu)

    if on_ipu:
        optimizer = poptorch.optim.Adam(model.parameters(),
                                        lr=hparams["training"]["learning_rate"],
                                        weight_decay=hparams["training"]["weight_decay"])
        model = poptorch.trainingModel(model, optimizer=optimizer)
        compile_ipu_model(model, train_loader)
        device = "ipu"
        print("Training on IPU")
    else:
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=hparams["training"]["learning_rate"],
                                     weight_decay=hparams["training"]["weight_decay"])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        print("Training on", device)

    history = {"train_loss": [], "val_loss": [], "model_params": [], "optim_params": []}

    for epoch in range(hparams["training"]["num_epochs"]):
        train_loss = train_step(model, train_loader, optimizer, device)
        val_loss = val_step(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["model_params"].append(copy.deepcopy(model.state_dict()))
        history["optim_params"].append(copy.deepcopy(optimizer.state_dict()))

        print("[Epoch {0}/{1}] Train: {2:.4f} | Val {3:.4f}".format(
            epoch + 1, hparams['training']['num_epochs'], train_loss, val_loss
        ))

    if on_ipu:
        model.detachFromDevice()

    if not save_dir.endswith("/"):
        save_dir = save_dir + "/"
    os.makedirs(save_dir, exist_ok=True)
    torch.save({
        "history": history,
        "hparams": hparams
    }, datetime.now().strftime(save_dir + "%Y-%m-%d_%H-%M-%S.run"))
    return history


def compile_ipu_model(model, loader):
    data = loader.dataset[0]
    fake_x = data.x.repeat(loader.batch_size, 1)
    fake_y = data.y.repeat(loader.batch_size, 1)
    fake_idx = data.edge_index.repeat(1, loader.batch_size)
    model.compile(fake_x, fake_idx, fake_y)


def evaluate_mse_nse(model, dataset, on_ipu=False):
    if on_ipu:
        device = "ipu"
        model = poptorch.inferenceModel(model)
        model.compile(dataset[0].x, dataset[0].edge_index)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
    model.eval()
    node_mses = torch.zeros(dataset[0].num_nodes, 1)
    with torch.no_grad():
        for data in tqdm(dataset, desc="Testing"):
            if device != "ipu":
                data = data.to(device)
            pred = model(data.x, data.edge_index)
            node_mses += mse_loss(pred, data.y, reduction="none") / len(dataset)
    if on_ipu:
        model.detachFromDevice()
    if dataset.normalized:
        node_mses *= dataset.std.square()
    nose_nses = 1 - node_mses / dataset.std.square()
    return node_mses, nose_nses

def evaluate_directory(chkpt_dir, eval_func, readout_func):
    results = {}
    for file in os.listdir(chkpt_dir):
        try:
            chkpt = torch.load(chkpt_dir + "/" + file)
        except:
            continue
        model, dataset = load_model_and_dataset(chkpt)
        results[readout_func(chkpt)] = eval_func(model, dataset)
    return results


def dirichlet_energy(x, edge_index, edge_weight, normalization=None):
    edge_index, edge_weight = to_undirected(edge_index, edge_weight)
    edge_index, edge_weight = get_laplacian(edge_index, edge_weight, normalization=normalization)
    lap = to_torch_coo_tensor(edge_index=edge_index, edge_attr=edge_weight)
    return 0.5 * torch.trace(torch.mm(x.T, torch.sparse.mm(lap, x)))


def evaluate_dirichlet_energy(model, dataset, on_ipu=False):
    if on_ipu:
        device = "ipu"
        model = poptorch.inferenceModel(model)
        model.compile(dataset[0].x, dataset[0].edge_index, evo_tracking=True)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
    model.eval()
    dirichlet_stats = []
    with torch.no_grad():
        edge_weights = model.edge_weights.detach().nan_to_num()
        for data in tqdm(dataset, desc="Testing"):
            if device != "ipu":
                data = data.to(device)
            _, evo = model(data.x, data.edge_index, evo_tracking=True)
            dir_energies = torch.tensor([dirichlet_energy(h, data.edge_index, edge_weights) for h in evo])
            dirichlet_stats.append(dir_energies)
    dirichlet_stats = torch.stack(dirichlet_stats)
    if on_ipu:
        model.detachFromDevice()
    return dirichlet_stats


def rolling_forecast(model, data, num_steps):
    window = data.x
    with torch.no_grad():
        for i in range(num_steps):
            pred = model(window[:, i:], data.edge_index)
            window = torch.cat([window, pred], dim=1)
    return window
