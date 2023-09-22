import copy
import numpy as np
import os
import random
import torch
import torch.nn as nn

from dataset import LamaHDataset
from models import MLP, GCN, ResGCN, GCNII
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
        return MLP(in_channels=hparams["data"]["window_size"] * (1 + len(dataset.MET_COLS)),
                   hidden_channels=hparams["model"]["hidden_channels"],
                   num_hidden=hparams["model"]["num_layers"],
                   param_sharing=hparams["model"]["param_sharing"])
    elif model_arch == "GCN":
        return GCN(in_channels=hparams["data"]["window_size"] * (1 + len(dataset.MET_COLS)),
                   hidden_channels=hparams["model"]["hidden_channels"],
                   num_hidden=hparams["model"]["num_layers"],
                   param_sharing=hparams["model"]["param_sharing"],
                   edge_orientation=hparams["model"]["edge_orientation"],
                   edge_weights=edge_weights
                   )
    elif model_arch == "ResGCN":
        return ResGCN(in_channels=hparams["data"]["window_size"] * (1 + len(dataset.MET_COLS)),
                      hidden_channels=hparams["model"]["hidden_channels"],
                      num_hidden=hparams["model"]["num_layers"],
                      param_sharing=hparams["model"]["param_sharing"],
                      edge_orientation=hparams["model"]["edge_orientation"],
                      edge_weights=edge_weights)
    elif model_arch == "GCNII":
        return GCNII(in_channels=hparams["data"]["window_size"] * (1 + len(dataset.MET_COLS)),
                     hidden_channels=hparams["model"]["hidden_channels"],
                     num_hidden=hparams["model"]["num_layers"],
                     param_sharing=hparams["model"]["param_sharing"],
                     edge_orientation=hparams["model"]["edge_orientation"],
                     edge_weights=edge_weights)
    raise ValueError("unknown model architecture", model_arch)


def load_dataset(path, hparams, split):
    if split == "train":
        years = hparams["training"]["train_years"]
    elif split == "test":
        years = set(range(2000, 2018)) - set(hparams["training"]["train_years"])
    else:
        raise ValueError("unknown split", split)
    return LamaHDataset(path,
                        years=years,
                        window_size=hparams["data"]["window_size"],
                        stride_length=hparams["data"]["stride_length"],
                        lead_time=hparams["data"]["lead_time"],
                        normalized=hparams["data"]["normalized"])


def load_model_and_dataset(chkpt, dataset_path):
    model_params = chkpt["history"]["best_model_params"]
    dataset = load_dataset(dataset_path, chkpt["hparams"], split="test")
    model = construct_model(chkpt["hparams"], dataset)
    model.load_state_dict(model_params, strict=False)
    return model, dataset


def train_step(model, train_loader, optimizer, device):
    model.train()
    train_loss = 0.0
    for batch in tqdm(train_loader, desc="Training"):
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)
        loss = mse_loss(out, batch.y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch.num_graphs / len(train_loader.dataset)
    return train_loss


def val_step(model, val_loader, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)
            loss = mse_loss(out, batch.y)
            val_loss += loss.item() * batch.num_graphs / len(val_loader.dataset)
    return val_loss


def train(model, dataset, hparams, early_stopping_patience=5):
    print(summary(model, depth=2))

    holdout_size = hparams["training"]["holdout_size"]
    train_dataset, val_dataset = random_split(dataset, [1 - holdout_size, holdout_size])
    train_loader = DataLoader(train_dataset, batch_size=hparams["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=hparams["training"]["batch_size"], shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=hparams["training"]["learning_rate"],
                                 weight_decay=hparams["training"]["weight_decay"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print("Training on", device)

    history = {"train_loss": [], "val_loss": [], "best_model_params": None}

    min_val_loss = float("inf")
    impatience_counter = 0
    for epoch in range(hparams["training"]["num_epochs"]):
        train_loss = train_step(model, train_loader, optimizer, device)
        val_loss = val_step(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print("[Epoch {0}/{1}] Train: {2:.4f} | Val {3:.4f}".format(
            epoch + 1, hparams["training"]["num_epochs"], train_loss, val_loss
        ))

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            impatience_counter = 0
            history["best_model_params"] = copy.deepcopy(model.state_dict())
        else:
            impatience_counter += 1
            if impatience_counter >= early_stopping_patience:
                print(f"Stopping early as val loss did not improve {early_stopping_patience} consecutive times")
                break

    return history


def save_checkpoint(history, hparams, filename, directory="./runs"):
    directory = directory.rstrip("/")
    os.makedirs(directory, exist_ok=True)
    out_path = f"{directory}/{filename}"
    torch.save({
        "history": history,
        "hparams": hparams
    }, out_path)
    print("Saved checkpoint", out_path)


def load_checkpoint(chkpt_path):
    return torch.load(chkpt_path)


def k_fold_cross_validation_split(index, k):
    index = list(index)
    random.shuffle(index)

    return [([j for i, j in enumerate(index) if i % k != fold],
             [j for i, j in enumerate(index) if i % k == fold])
            for fold in range(k)]


def evaluate_mse_nse(model, dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    node_mses = torch.zeros(dataset[0].num_nodes, 1)
    with torch.no_grad():
        for data in tqdm(dataset, desc="Testing"):
            data = data.to(device)
            pred = model(data.x, data.edge_index)
            node_mses += mse_loss(pred, data.y, reduction="none").cpu() / len(dataset)
    sigma_squared = dataset.std[:, [0]].square()
    if dataset.normalized:
        node_mses *= sigma_squared
    nose_nses = 1 - node_mses / sigma_squared
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


def evaluate_dirichlet_energy(model, dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    dirichlet_stats = []
    with torch.no_grad():
        edge_weights = model.edge_weights.detach().nan_to_num()
        for data in tqdm(dataset, desc="Testing"):
            data = data.to(device)
            _, evo = model(data.x, data.edge_index, evo_tracking=True)
            dir_energies = torch.tensor([dirichlet_energy(h, data.edge_index, edge_weights) for h in evo])
            dirichlet_stats.append(dir_energies)
    dirichlet_stats = torch.stack(dirichlet_stats)
    return dirichlet_stats


def rolling_forecast(model, data, num_steps):
    window = data.x
    with torch.no_grad():
        for i in range(num_steps):
            pred = model(window[:, i:], data.edge_index)
            window = torch.cat([window, pred], dim=1)
    return window
