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
        "batch_size": 64,
        "learning_rate": 1e-4,
        "weight_decay": 0,
        "random_seed": 42,
        "train_years": None,  # set below
        "holdout_size": 0.25,
    }
}


functions.ensure_reproducibility(hparams["training"]["random_seed"])

for fold, (train_years, test_years) in enumerate(functions.k_fold_cross_validation_split(range(2000, 2018), k=6)):
    for architecture in ["GCN", "ResGCN", "GCNII", "GRAFFNN"]:
        for edge_orientation in ["downstream", "upstream", "bidirectional"]:
            for adjacency_type in ["isolated", "binary", "stream_length", "elevation_difference", "average_slope", "learned"]:
                hparams["training"]["train_years"] = train_years
                hparams["model"]["architecture"] = architecture
                hparams["model"]["edge_orientation"] = edge_orientation
                hparams["model"]["adjacency_type"] = adjacency_type

                dataset = functions.load_dataset(hparams, "train")
                model = functions.construct_model(hparams, dataset)
                history = functions.train(model, dataset, hparams)
            
                functions.save_checkpoint(history, hparams,
                                          f"{architecture}_{edge_orientation}_{adjacency_type}_{fold}.run",
                                          directory="./runs/topology")
