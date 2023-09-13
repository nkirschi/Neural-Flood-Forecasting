import functions
import torch
import pandas as pd

from datetime import datetime

OUT_FILE = f"results_{datetime.now()}.txt"

results_string = ""
for num_layers in range(1, 21):
    print(num_layers, "layers")
    summary_df = pd.DataFrame()
    fold_results = []
    for fold in range(6):
        chkpt = functions.load_checkpoint(f"/scratch/kirschstein/runs/depth/layers{num_layers:02d}_{fold}.run")
        model, dataset = functions.load_model_and_dataset(chkpt)
        test_mse, test_nse = functions.evaluate_mse_nse(model, dataset)
        summary_df.loc[fold, "mean_mse"] = test_mse.mean().item()
        summary_df.loc[fold, "mean_nse"] = test_nse.mean().item()
        fold_results.append(torch.cat([test_mse, test_nse], dim=1))
    with open(OUT_FILE, "a") as f:
        f.write(f"{num_layers} layers\n")
        f.write(str(pd.DataFrame(torch.stack(fold_results).mean(dim=0)).describe()) + "\n")
        f.write(str(pd.DataFrame(summary_df).describe()) + "\n\n")

with open(OUT_FILE, "r") as f:
    s = f.read()
print(s)
