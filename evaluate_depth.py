import functions
import pandas as pd

from datetime import datetime

DATASET_PATH = "/path/to/LamaH-CE"
CHECKPOINT_PATH = "./checkpoints/depth"
OUT_FILE = f"results/results_depth_{datetime.now()}.csv"

results_df = pd.DataFrame()
for architecture in ["ResGCN", "GCNII"]:
    for num_layers in range(1, 21):
        for fold in range(6):
            run_id = f"{architecture}_bidirectional_binary_depth{num_layers:02d}_{fold}"
            print(run_id)
            chkpt = functions.load_checkpoint(f"{CHECKPOINT_PATH}/{run_id}.run")
            model, dataset = functions.load_model_and_dataset(chkpt, DATASET_PATH)
            test_mse, test_nse = functions.evaluate_nse(model, dataset)
            results_df.loc[run_id + "_MSE", range(len(test_mse))] = test_mse.squeeze().numpy()
            results_df.loc[run_id + "_NSE", range(len(test_mse))] = test_nse.squeeze().numpy()
results_df.to_csv(OUT_FILE)
