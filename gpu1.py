import functions

hparams = {
    "data": {
        "window_size": 24,
        "stride_length": 1,
        "lead_time": 6,
        "normalized": True,
    },
    "model": {
        "architecture": "ResGCN",
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

DATASET_PATH = "/scratch/kirschstein/LamaH-CE"
CHECKPOINT_PATH = "./runs/depth"

for fold, (train_years, test_years) in enumerate([([2004, 2015, 2016, 2008, 2014, 2011, 2010, 2013, 2002, 2000, 2017, 2005, 2009, 2007, 2003], [2006, 2012, 2001]),
([2004, 2015, 2016, 2006, 2014, 2011, 2010, 2013, 2012, 2000, 2017, 2005, 2009, 2001, 2003], [2008, 2002, 2007]),
([2004, 2015, 2016, 2006, 2008, 2011, 2010, 2013, 2012, 2002, 2017, 2005, 2009, 2001, 2007], [2014, 2000, 2003])]):
    for num_layers in range(1, 21):
        hparams["training"]["train_years"] = train_years
        hparams["model"]["num_layers"] = num_layers

        functions.ensure_reproducibility(hparams["training"]["random_seed"])

        dataset = functions.load_dataset(DATASET_PATH, hparams, split="train")
        model = functions.construct_model(hparams, dataset)
        history = functions.train(model, dataset, hparams)

        functions.save_checkpoint(history, hparams,
                                  f"layers{num_layers:02d}_{fold}.run",
                                  directory=CHECKPOINT_PATH)
