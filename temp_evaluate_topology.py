import functions
import pandas as pd

from datetime import datetime

DATASET_PATH = "/scratch/kirschstein/LamaH-CE"
CHECKPOINT_PATH = "/scratch/kirschstein/runs/topology_temp"
OUT_FILE = f"results/results_topology_{datetime.now()}.csv"

results_df = pd.DataFrame()
for architecture in ["GCNII"]:
    for edge_orientation in ["bidirectional"]:
        for adjacency_type in ["binary", "learned"]:
            for fold in range(3):
                run_id = f"{architecture}_{edge_orientation}_{adjacency_type}_{fold}"
                print(run_id)
                chkpt = functions.load_checkpoint(f"{CHECKPOINT_PATH}/{run_id}.run")
                model, dataset = functions.load_model_and_dataset(chkpt, DATASET_PATH)
                test_nse = functions.evaluate_mse_nse(model, dataset)
                results_df.loc[run_id] = test_nse.squeeze().numpy()
results_df.to_csv(OUT_FILE)
