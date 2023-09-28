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
        "num_layers": None,  # set below
        "hidden_channels": 128,
        "param_sharing": False,
        "edge_orientation": "bidirectional",
        "adjacency_type": "binary",
    },
    "training": {
        "num_epochs": 30,
        "batch_size": 64,
        "learning_rate": 5e-4,
        "weight_decay": 0,
        "random_seed": 42,
        "train_years": None,  # set below
        "holdout_size": 3/18 * 15/18,
    }
}

DATASET_PATH = "/path/to/LamaH-CE"
CHECKPOINT_PATH = "./checkpoints/depth"

for fold, (train_years, test_years) in enumerate(functions.k_fold_cross_validation_split(range(2000, 2018), k=6)):
    for architecture in ["ResGCN", "GCNII"]:
        for num_layers in range(1, 21):
            hparams["training"]["train_years"] = train_years
            hparams["model"]["architecture"] = architecture
            hparams["model"]["num_layers"] = num_layers

            functions.ensure_reproducibility(hparams["training"]["random_seed"])

            dataset = functions.load_dataset(DATASET_PATH, hparams, split="train")
            model = functions.construct_model(hparams, dataset)
            history = functions.train(model, dataset, hparams)

            functions.save_checkpoint(history, hparams,
                                      f"{architecture}_bidirectional_binary_depth{num_layers:02d}_{fold}.run",
                                      directory=CHECKPOINT_PATH)
