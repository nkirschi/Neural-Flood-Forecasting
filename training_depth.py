import functions

hparams = {
    "data": {
        "window_size": 24,
        "stride_length": 1,
        "lead_time": 1,
        "normalized": True
    },
    "model": {
        "architecture": "ResGCN",
        "num_layers": None,  # set below
        "hidden_channels": 64,
        "param_sharing": False,
        "graff_step_size": 1,
        "edge_orientation": "bidirectional",
        "adjacency_type": "binary"
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

for fold, (train_years, test_years) in enumerate(functions.k_fold_cross_validation_split(range(2000, 2018), k=6)):
    for num_layers in range(1, 21):
        hparams["training"]["train_years"] = train_years
        hparams["model"]["num_layers"] = num_layers

        functions.ensure_reproducibility(hparams["training"]["random_seed"])

        dataset = functions.load_dataset(hparams, "train")
        model = functions.construct_model(hparams, dataset)
        history = functions.train(model, dataset, hparams)

        functions.save_checkpoint(history, hparams,
                                  f"layers{num_layers:02d}_{fold}.run",
                                  directory="./runs/depth")
