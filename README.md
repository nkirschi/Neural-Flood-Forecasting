# The Merit of River Network Topology for Neural Flood Forecasting

This is the code repository for the ICML paper [The Merit of River Network Topology for Neural Flood Forecasting](https://dl.acm.org/doi/10.5555/3692070.3693060).

## Citation

If you use the code or models in this repository for your research, please cite our paper:
```
@inproceedings{10.5555/3692070.3693060,
    author = {Kirschstein, Nikolas and Sun, Yixuan},
    title = {The merit of river network topology for neural flood forecasting},
    year = {2024},
    publisher = {JMLR.org},
    booktitle = {Proceedings of the 41st International Conference on Machine Learning},
    location = {Vienna, Austria},
    series = {ICML'24}
}
```

## Installation

This project is written in pure Python. To deploy it on your machine, use

```
pip install -e .
```

in the repository's root directory. While the development mode -e is not strictly necessary, we recommend it since it ensures any modifications to the code take effect immediately.


## Scripts

No environment variables are used for our scripts. Instead, the 

### Training

To train models, use one of the following scripts in the root directory. You need to specify the hyperparameters in the `hparams` dict and adjust the paths to the base directory of the dataset in `DATASET_PATH` and desired base directory for model checkpoints in `CHECKPOINT_PATH`.

- `train_full.py`: Training on the full river network.
- `train_mlp.py`: Training an MLP on the full river network as a baseline.
- `train_ablation.py`: Training on the full river network for varying window sizes and lead times.
- `train_subnetworks.py`: Training on the small sub-graphs discussed in the paper.

After execution, there will be a folder structure underneath `checkpoints/` that is implied by the `chkpt_name` pattern in the respective script.


### Testing

To test models obtained as per the previous section, use the respective script with a `test_` prefix instead of `train_`. You need to specify the path to the base directory of the dataset in `DATASET_PATH`, the path to the base directory of the saved checkpoints in `CHECKPOINT_PATH`, and the path for the result summary file in `RESULTS_FILE`.
After execution, there will be a folder with CSV files containing the test results. Those can be processed with the pandas library; see `notebook_topology_comparison.ipynb` for an example.


### Modelling

If you seek to make changes to the methodology, the relevant source files are:

- `datsaset.py`: Definition of the LamaH-CE dataset class.
- `models.py`: Definition of the discharge prediction models.
- `functions.py`: Definition of train/test logic and helper functions.

## Notebooks

All files with the `notebook_` prefix are Jupyter notebooks that extract interesting insights from the results and checkpoints. You need to install Jupyter Lab in addition to the `requirements.txt` if you wish to execute the notebooks.