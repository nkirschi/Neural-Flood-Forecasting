import functions
import pandas as pd

DATASET_PATH = "/scratch/kirschstein/LamaH-CE"
CHECKPOINT_PATH = "/scratch/kirschstein/runs/ablation"
OUT_FILE_UNWEIGHTED = "~/floodgnn/results/results_ablation_unweighted.csv"
OUT_FILE_WEIGHTED = "~/floodgnn/results/results_ablation_weighted.csv"

unweighted_df = pd.DataFrame()
weighted_df = pd.DataFrame()
for root_gauge_id in [71, 211, 387, 532]:
    for window_size in [72, 60, 48, 36, 24, 12]:
        for lead_time in [12, 9, 6, 3, 2, 1]:
            for fold_id in range(3):
                run_id = f"{root_gauge_id}_{window_size}_{lead_time}_{fold_id}"
                print(run_id)
                chkpt = functions.load_checkpoint(f"{CHECKPOINT_PATH}/{run_id}.run")
                model, dataset = functions.load_model_and_dataset(chkpt, DATASET_PATH)
                unweighted_nse, weighted_nse = functions.evaluate_nse(model, dataset)
                unweighted_df.loc[run_id, range(len(unweighted_nse))] = unweighted_nse.squeeze().numpy()
                weighted_df.loc[run_id, range(len(weighted_nse))] = weighted_nse.squeeze().numpy()
unweighted_df.to_csv(OUT_FILE_UNWEIGHTED)
weighted_df.to_csv(OUT_FILE_WEIGHTED)
