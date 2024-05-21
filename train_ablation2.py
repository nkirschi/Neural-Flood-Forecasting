import functions

hparams = {
    "data": {
        "root_gauge_id": 399,
        "rewire_graph": True,
        "window_size": None,  # set below
        "stride_length": 1,
        "lead_time": None,  # set below
        "normalized": True,
    },
    "model": {
        "architecture": "GCNII",
        "num_layers": None,  # set below
        "hidden_channels": 128,
        "param_sharing": False,
        "edge_orientation": "bidirectional",
        "adjacency_type": "learned",
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
CHECKPOINT_PATH = "/scratch/kirschstein/runs/ablation2"

for fold_id, (train_years, test_years) in enumerate([(list(range(2000, 2016, 2)), [2016, 2017]),
                                                     (list(range(2001, 2016, 2)), [2016, 2017]),
                                                     (list(range(2008, 2016, 1)), [2016, 2017])]):
    for window_size in [72, 60, 48, 36, 24, 12]:
        for lead_time in [12, 9, 6, 3, 2, 1]:
            hparams["training"]["train_years"] = train_years
            hparams["data"]["window_size"] = window_size
            hparams["data"]["lead_time"] = lead_time
            dataset = functions.load_dataset(DATASET_PATH, hparams, split="train")
            hparams["model"]["num_layers"] = dataset.longest_path()

            functions.ensure_reproducibility(hparams["training"]["random_seed"])

            print(hparams["model"]["num_layers"], "layers used")
            model = functions.construct_model(hparams, dataset)
            history = functions.train(model, dataset, hparams)

            chkpt_name = f"{window_size}_{lead_time}_{fold_id}.run"
            functions.save_checkpoint(history, hparams, chkpt_name, directory=CHECKPOINT_PATH)
