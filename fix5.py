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
        "num_epochs": 30,
        "batch_size": 64,
        "learning_rate": 5e-4,
        "weight_decay": 0,
        "random_seed": 42,
        "train_years": None,  # set below
        "holdout_size": 3/18 * 15/18,
    }
}

DATASET_PATH = "/scratch/kirschstein/LamaH-CE"
CHECKPOINT_PATH = "./runs/topology_fix"

for fold, (train_years, test_years) in [(3, (
                                        [2004, 2015, 2016, 2008, 2014, 2011, 2010, 2013, 2002, 2000, 2017, 2005, 2009,
                                         2007, 2003], [2006, 2012, 2001])),
                                        (4, (
                                        [2004, 2015, 2016, 2006, 2014, 2011, 2010, 2013, 2012, 2000, 2017, 2005, 2009,
                                         2001, 2003], [2008, 2002, 2007])),
                                        (5, (
                                        [2004, 2015, 2016, 2006, 2008, 2011, 2010, 2013, 2012, 2002, 2017, 2005, 2009,
                                         2001, 2007], [2014, 2000, 2003]))
                                        ]:
    for architecture in ["ResGCN"]:
        for edge_orientation in ["bidirectional"]:
            for adjacency_type in ["binary"]:
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
