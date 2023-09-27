import functions
import pandas as pd

from datetime import datetime

DATASET_PATH = "/scratch/kirschstein/LamaH-CE"
CHECKPOINT_DIR = "./runs/topology"
OUT_FILE = f"results_topology_mlp.csv"

results_df = pd.DataFrame()
for fold in range(6):
    run_id = f"MLP_{fold}"
    print(run_id)
    chkpt = functions.load_checkpoint(f"{CHECKPOINT_DIR}/{run_id}.run")
    model, dataset = functions.load_model_and_dataset(chkpt, DATASET_PATH)
    test_mse, test_nse = functions.evaluate_mse_nse(model, dataset)
    results_df.loc[run_id + "_MSE", range(len(test_mse))] = test_mse.squeeze().numpy()
    results_df.loc[run_id + "_NSE", range(len(test_mse))] = test_nse.squeeze().numpy()
results_df.to_csv(OUT_FILE)
