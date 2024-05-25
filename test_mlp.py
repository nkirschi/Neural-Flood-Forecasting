import functions
import pandas as pd

DATASET_PATH = "/path/to/LamaH-CE"
CHECKPOINT_PATH = "/path/to/checkpoints/mlp"
OUT_FILE = "results/results_mlp.csv"

results_df = pd.DataFrame()
for fold in range(3):
    run_id = f"MLP_{fold}"
    print(run_id)
    chkpt = functions.load_checkpoint(f"{CHECKPOINT_PATH}/{run_id}.run")
    model, dataset = functions.load_model_and_dataset(chkpt, DATASET_PATH)
    test_nse = functions.evaluate_nse(model, dataset)
    results_df.loc[run_id, range(len(test_nse))] = test_nse.squeeze().numpy()
results_df.to_csv(OUT_FILE)
