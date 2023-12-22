import functions
import pandas as pd

from datetime import datetime

DATASET_PATH = "/scratch/kirschstein/LamaH-CE"
CHECKPOINT_PATH = "/scratch/kirschstein/runs/gadgets"
OUT_FILE = f"results/results_gadgets_{datetime.now()}.csv"

results_df = pd.DataFrame()
for root_gauge_id in [71, 211, 387]:
    for architecture in ["ResGCN", "GCNII"]:  # ["GCN", "ResGCN", "GCNII"]:
        for edge_orientation in ["downstream", "upstream", "bidirectional"]:
            for adjacency_type in ["isolated", "binary", "stream_length", "elevation_difference", "average_slope", "learned"]:
                for fold_id in range(3):
                    run_id = f"{root_gauge_id}_{architecture}_{edge_orientation}_{adjacency_type}_{fold_id}"
                    print(run_id)
                    chkpt = functions.load_checkpoint(f"{CHECKPOINT_PATH}/{run_id}.run")
                    model, dataset = functions.load_model_and_dataset(chkpt, DATASET_PATH)
                    test_mse, test_nse = functions.evaluate_mse_nse(model, dataset)
                    results_df.loc[run_id + "_MSE", range(len(test_mse))] = test_mse.squeeze().numpy()
                    results_df.loc[run_id + "_NSE", range(len(test_mse))] = test_nse.squeeze().numpy()
results_df.to_csv(OUT_FILE)
