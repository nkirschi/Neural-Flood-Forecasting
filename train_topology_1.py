import functions

hparams = {
    "data": {
        "window_size": 24,
        "stride_length": 1,
        "lead_time": 6,
        "normalized": True,
    },
    "model": {
        "architecture": None,  # set below
        "num_layers": 20,
        "hidden_channels": 128,
        "param_sharing": False,
        "edge_orientation": None,  # set below
        "adjacency_type": None,  # set below
    },
    "training": {
        "num_epochs": 20,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "random_seed": 42,
        "train_years": None,  # set below
        "holdout_size": 1/5,
    }
}

DATASET_PATH = "/scratch/kirschstein/LamaH-CE"
CHECKPOINT_PATH = "/scratch/kirschstein/runs/topology"

for fold, (train_years, test_years) in [(1, ([2001, 2004, 2007, 2010, 2013], [2015, 2017]))]:
# for fold, (train_years, test_years) in enumerate(functions.k_fold_cross_validation_split(range(2000, 2018), k=6)):
    for architecture in ["ResGCN"]:
        for edge_orientation in ["downstream", "upstream", "bidirectional"]:
            for adjacency_type in ["isolated", "binary", "stream_length", "elevation_difference", "average_slope", "learned"]:
                hparams["training"]["train_years"] = train_years
                hparams["model"]["architecture"] = architecture
                hparams["model"]["edge_orientation"] = edge_orientation
                hparams["model"]["adjacency_type"] = adjacency_type

                functions.ensure_reproducibility(hparams["training"]["random_seed"])

                dataset = functions.load_dataset(DATASET_PATH, hparams, split="train")
                model = functions.construct_model(hparams, dataset)
                history = functions.train(model, dataset, hparams)

                functions.save_checkpoint(history, hparams,
                                          f"{architecture}_{edge_orientation}_{adjacency_type}_{fold}.run",
                                          directory=CHECKPOINT_PATH)
