import functions
import torch
import pandas as pd

from datetime import datetime

DATASET_PATH = "/scratch/kirschstein2/LamaH-CE"
CHECKPOINT_DIR = "./runs/topology"
OUT_FILE = f"results_{datetime.now()}.txt"

results_string = ""
for architecture in ["GCN", "ResGCN", "GCNII"]:
    for edge_orientation in ["downstream", "upstream", "bidirectional"]:
        for adjacency_type in ["isolated", "binary", "stream_length", "elevation_difference", "average_slope", "learned"]:
            print(architecture, edge_orientation, adjacency_type)
            summary_df = pd.DataFrame()
            fold_results = []
            for fold in range(4, 6):
                chkpt = functions.load_checkpoint(f"{CHECKPOINT_DIR}/{architecture}_{edge_orientation}_{adjacency_type}_{fold}.run")
                model, dataset = functions.load_model_and_dataset(chkpt, DATASET_PATH)
                test_mse, test_nse = functions.evaluate_mse_nse(model, dataset)
                summary_df.loc[fold, "mean_mse"] = test_mse.mean().item()
                summary_df.loc[fold, "mean_nse"] = test_nse.mean().item()
                fold_results.append(torch.cat([test_mse, test_nse], dim=1))
            with open(OUT_FILE, "a") as f:
                f.write(f"{architecture}_{edge_orientation}_{adjacency_type}\n")
                f.write(str(pd.DataFrame(torch.stack(fold_results).mean(dim=0)).describe()) + "\n")
                f.write(str(pd.DataFrame(summary_df).describe()) + "\n\n")

with open(OUT_FILE, "r") as f:
    s = f.read()
print(s)
