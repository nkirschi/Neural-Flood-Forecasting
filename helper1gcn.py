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
        "learning_rate": 5e-4,
        "weight_decay": 0,
        "random_seed": 42,
        "train_years": None,  # set below
        "holdout_size": 3/18 * 15/18,
    }
}

DATASET_PATH = "/scratch/kirschstein2/LamaH-CE"
CHECKPOINT_PATH = "./runs/topology_helper"

for fold, (train_years, test_years) in [(1, ([2004, 2016, 2006, 2008, 2014, 2011, 2013, 2012, 2002, 2000, 2017, 2009, 2001, 2007, 2003], [2015, 2010, 2005]))]:
    hparams["training"]["train_years"] = train_years
    dataset = functions.load_dataset(DATASET_PATH, hparams, split="train")
    for architecture in ["GCN"]:
        for edge_orientation in ["downstream", "upstream", "bidirectional"]:
            for adjacency_type in ["isolated", "binary", "stream_length", "elevation_difference", "average_slope", "learned"]:
                hparams["model"]["architecture"] = architecture
                hparams["model"]["edge_orientation"] = edge_orientation
                hparams["model"]["adjacency_type"] = adjacency_type

                functions.ensure_reproducibility(hparams["training"]["random_seed"])

                model = functions.construct_model(hparams, dataset)
                history = functions.train(model, dataset, hparams)

                functions.save_checkpoint(history, hparams,
                                          f"{architecture}_{edge_orientation}_{adjacency_type}_{fold}.run",
                                          directory=CHECKPOINT_PATH)
