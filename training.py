import functions

hparams = {
    "data": {
        "window_size": 24,
        "stride_length": 1,
        "lead_time": 1,
        "normalized": True
    },
    "model": {
        "architecture": None,  # set below
        "num_layers": 20,
        "hidden_channels": 64,
        "param_sharing": False,
        "graff_step_size": 1,
        "edge_orientation": None,  # set below
        "adjacency_type": None  # set below
    },
    "training": {
        "num_epochs": 20,
        "batch_size": 16,
        "learning_rate": 1e-4,
        "weight_decay": 0,
        "random_seed": None,  # set below
        "holdout_size": 0.25
    }
}

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

for architecture in ["GCN", "ResGCN", "GCNII", "GRAFFNN"]:
    for edge_orientation in ["downstream", "upstream", "bidirectional"]:
        for adjacency_type in ["isolated", "binary", "stream_length", "elevation_difference", "average_slope", "learned"]:
            for random_seed in range(5):
                hparams["model"]["architecture"] = architecture
                hparams["model"]["edge_orientation"] = edge_orientation
                hparams["model"]["adjacency_type"] = adjacency_type
                hparams["training"]["random_seed"] = random_seed
            
                functions.ensure_reproducibility(hparams["training"]["random_seed"])
                dataset = functions.load_dataset(hparams, "train")
                model = functions.construct_model(hparams, dataset)
                history = functions.train(model, dataset, hparams)
            
                functions.save_checkpoint({
                    "history": history,
                    "hparams": hparams
                }, f"{architecture}_{edge_orientation}_{adjacency_type}_{random_seed}.run")
