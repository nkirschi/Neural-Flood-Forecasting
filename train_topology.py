import functions

hparams = {
    "data": {
        "base_gauge_id": 71, # 399,
        "rewire_graph": False, # True,
        "window_size": 24,
        "stride_length": 1,
        "lead_time": 6,
        "normalized": True,
    },
    "model": {
        "architecture": None,  # set below
        "num_layers": None,  # set below
        "hidden_channels": 128,
        "param_sharing": False,
        "edge_orientation": None,  # set below
        "adjacency_type": None,  # set below
    },
    "training": {
        "num_epochs": 50,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "weight_decay": 5e-4,
        "random_seed": 42,
        "train_years": None,  # set below
        "holdout_size": 1/5,
    }
}

DATASET_PATH = "/scratch/kirschstein/LamaH-CE"
CHECKPOINT_PATH = "/scratch/kirschstein/runs/topology"

for fold_id, (train_years, test_years) in enumerate([(list(range(2000, 2016, 2)), [2016, 2017]),
                                                     (list(range(2001, 2016, 2)), [2016, 2017]),
                                                     (list(range(2008, 2016, 1)), [2016, 2017])]):
    for architecture in ["ResGCN"]:  # ["GCN", "ResGCN", "GCNII"]
        for edge_orientation in ["bidirectional"]:  # ["downstream", "upstream", "bidirectional"]
            for adjacency_type in ["isolated", "binary"]:  # ["isolated", "binary", "stream_length", "elevation_difference", "average_slope", "learned"]
                hparams["training"]["train_years"] = train_years
                hparams["model"]["architecture"] = architecture
                hparams["model"]["edge_orientation"] = edge_orientation
                hparams["model"]["adjacency_type"] = adjacency_type

                functions.ensure_reproducibility(hparams["training"]["random_seed"])

                dataset = functions.load_dataset(DATASET_PATH, hparams, split="train")
                print(len(dataset))
                hparams["model"]["num_layers"] = dataset.longest_path()
                print(hparams["model"]["num_layers"], "layers used")
                model = functions.construct_model(hparams, dataset)
                history = functions.train(model, dataset, hparams)

                functions.save_checkpoint(history, hparams,
                                          f"{architecture}_{edge_orientation}_{adjacency_type}_{fold_id}.run",
                                          directory=CHECKPOINT_PATH)
