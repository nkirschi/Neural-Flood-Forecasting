import functions
import pandas as pd

from datetime import datetime

DATASET_PATH = "/scratch/kirschstein/LamaH-CE"
CHECKPOINT_PATH = "/scratch/kirschstein/runs/mlp"
OUT_FILE = f"{CHECKPOINT_PATH}/results.csv"

results_df = pd.DataFrame()
for fold in range(3):
    run_id = f"MLP_{fold}"
    print(run_id)
    chkpt = functions.load_checkpoint(f"{CHECKPOINT_PATH}/{run_id}.run")
    model, dataset = functions.load_model_and_dataset(chkpt, DATASET_PATH)
    test_nse = functions.evaluate_nse(model, dataset)
    results_df.loc[run_id, range(len(test_nse))] = test_nse.squeeze().numpy()
results_df.to_csv(OUT_FILE)
