import functions
import pandas as pd

DATASET_PATH = "/path/to/LamaH-CE"
CHECKPOINT_PATH = "/path/to/checkpoints/ablation"
OUT_FILE = "results/results_ablation.csv"

unweighted_df = pd.DataFrame()
results_df = pd.DataFrame()
for window_size in [72, 60, 48, 36, 24, 12]:
    for lead_time in [12, 9, 6, 3, 2, 1]:
        for fold_id in range(3):
            run_id = f"{window_size}_{lead_time}_{fold_id}"
            print(run_id)
            chkpt = functions.load_checkpoint(f"{CHECKPOINT_PATH}/{run_id}.run")
            model, dataset = functions.load_model_and_dataset(chkpt, DATASET_PATH)
            test_nse = functions.evaluate_nse(model, dataset)
            results_df.loc[run_id, range(len(test_nse))] = test_nse.squeeze().numpy()
results_df.to_csv(OUT_FILE)
