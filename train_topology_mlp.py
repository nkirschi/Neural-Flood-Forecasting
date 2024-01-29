import functions

hparams = {
    "data": {
        "root_gauge_id": 399,
        "rewire_graph": True,
        "window_size": 24,
        "stride_length": 1,
        "lead_time": 6,
        "normalized": True,
    },
    "model": {
        "architecture": "MLP",
        "num_layers": None,  # set below
        "hidden_channels": 128,
        "param_sharing": False,
        "edge_orientation": "bidirectional",
        "adjacency_type": "isolated"
    },
    "training": {
        "num_epochs": 100,
        "batch_size": 64,
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
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
    hparams["training"]["train_years"] = train_years
    dataset = functions.load_dataset(DATASET_PATH, hparams, split="train")
    hparams["model"]["num_layers"] = dataset.longest_path()

    functions.ensure_reproducibility(hparams["training"]["random_seed"])

    print(hparams["model"]["num_layers"], "layers used")
    model = functions.construct_model(hparams, dataset)
    history = functions.train(model, dataset, hparams)

    chkpt_name = f"MLP_{fold_id}.run"
    functions.save_checkpoint(history, hparams, chkpt_name, directory=CHECKPOINT_PATH)
