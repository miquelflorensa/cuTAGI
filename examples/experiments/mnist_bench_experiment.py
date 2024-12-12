import subprocess
import itertools
import os
import sys
import torch
import wandb




def get_hyperparameter_combinations():
    """
    Define the hyperparameter grid and return all valid combinations.
    Only generate sigma_v combinations for the 'tagi' framework.
    For 'torch', sigma_v will always be None.
    """
    frameworks = ['tagi_remax', 'tagi_hrc', 'torch']
    models = {
        'torch': ['CNN', 'CNNBatchNorm', 'FNN'],
        'tagi_hrc': ['CNN', 'CNNBatchNorm', 'FNN'],
        'tagi_remax': ['CNN', 'CNNBatchNorm', 'FNN'],
    }

    num_epochs = [40]
    sigma_v = [0, 0.1, 0.5, 1.0]  # Only relevant for TAGI
    learning_rates = [1e-3]

    neurons_per_layer_fnn = [32, 512]
    channels_per_layer_cnn = [8, 32]

    batch_sizes = [512, 16]
    num_layers = [1, 3, 5]

    combinations = []
    for framework in frameworks:
        for model in models[framework]:
            for layer in num_layers:
                if model == "FNN":
                    for neurons in neurons_per_layer_fnn:
                        for batch in batch_sizes:
                            for epoch in num_epochs:
                                if framework == "torch":
                                    # Torch does not use sigma_v
                                    combinations.append(
                                        (framework, model, layer, neurons, None, batch, epoch, None, learning_rates[0])
                                    )
                                else:  # framework == 'tagi'
                                    for sv in sigma_v:
                                        combinations.append(
                                            (framework, model, layer, neurons, None, batch, epoch, sv, learning_rates[0])
                                        )
                elif model in ["CNN", "CNNBatchNorm"]:
                    for channels in channels_per_layer_cnn:
                        for batch in batch_sizes:
                            for epoch in num_epochs:
                                if framework == "torch":
                                    # Torch does not use sigma_v
                                    combinations.append(
                                        (framework, model, layer, None, channels, batch, epoch, None, learning_rates[0])
                                    )
                                else:  # framework == 'tagi'
                                    for sv in sigma_v:
                                        combinations.append(
                                            (framework, model, layer, None, channels, batch, epoch, sv, learning_rates[0])
                                        )
    return combinations




# Check if a run already exists in WandB
def check_existing_run(project_name, run_name):
    """
    Check if a run with the given name already exists in WandB.

    Args:
        project_name (str): The name of the WandB project.
        run_name (str): The name of the run to check.

    Returns:
        bool: True if the run exists, False otherwise.
    """
    api = wandb.Api()
    try:
        runs = api.runs(f"{wandb.api.default_entity}/{project_name}")
        print("runs : ", runs)
        print("run name: ", run_name)
        for run in runs:
            if run.name == run_name:
                print(f"Run '{run_name}' already exists in WandB. Skipping...")
                return True
    except Exception as e:
        print(f"Error querying WandB: {e}")
    return False

def run_experiment(
    framework, model, num_layers, neurons_per_layer, channels_per_layer,
    batch_size, num_epochs, sigma_v, learning_rate, project_name="Remax"
):
    """
    Run a single experiment by invoking mnist_bench.py with the specified parameters.
    Handle None values gracefully by substituting 0 where needed.
    """
    # For FNN models, channels are irrelevant; for CNN models, neurons are irrelevant.
    if model == "FNN":
        neurons_per_layer = neurons_per_layer or 0
        channels_per_layer = 0
    else:
        neurons_per_layer = 0
        channels_per_layer = channels_per_layer or 0

    # Construct run name
    sigma_str = f"sigma{sigma_v}" if sigma_v is not None else ""
    run_name = f"{framework}_{model}_layers{num_layers}"
    if model == "FNN":
        run_name += f"_neurons{neurons_per_layer}"
    else:
        run_name += f"_channels{channels_per_layer}"
    run_name += f"_batch{batch_size}"
    if sigma_str:
        run_name += f"_{sigma_str}"
    run_name += f"_lr{learning_rate}"

    print(f"\nChecking if run '{run_name}' exists in WandB...")

    # Skip if the run already exists in WandB
    if check_existing_run(project_name, run_name):
        return  # Skip this experiment


    print(f"\nRunning experiment: {run_name}")

    # Construct the command to run mnist_bench.py
    command = [
        sys.executable,
        "./examples/experiments/mnist_bench.py",
        "--framework", framework,
        "--model", model,
        "--num_layers", str(num_layers),
        "--batch_size", str(batch_size),
        "--epochs", str(num_epochs),
        "--device", "cuda" if torch.cuda.is_available() else "cpu",
        "--learning_rate", str(learning_rate)
    ]

    if model == "FNN":
        command.extend(["--neurons_per_layer", str(neurons_per_layer)])
    else:
        command.extend(["--channels_per_layer", str(channels_per_layer)])

    if sigma_v is not None:
        command.extend(["--sigma_v", str(sigma_v)])

    print(f"Executing command: {' '.join(command)}")

    try:
        subprocess.run(command, check=True)
        print(f"Experiment {run_name} completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Experiment {run_name} failed with error: {e}")

# Update your main function
def main():
    """
    Main function to execute the hyperparameter search.
    """
    # Ensure mnist_bench.py is in the same directory
    if not os.path.isfile("./examples/experiments/mnist_bench.py"):
        print("Error: mnist_bench.py not found in the current directory.")
        sys.exit(1)

    # Get all hyperparameter combinations
    hyperparameter_combinations = get_hyperparameter_combinations()
    total_runs = len(hyperparameter_combinations)
    print(f"Total experiments to run: {total_runs}")

    # Iterate and run experiments sequentially
    for idx, config in enumerate(hyperparameter_combinations, 1):
        (
            framework, model, num_layers, neurons_per_layer,
            channels_per_layer, batch_size, num_epochs,
            sigma_v, learning_rate
        ) = config

        print(f"\nStarting experiment {idx}/{total_runs}...")
        run_experiment(
            framework=framework,
            model=model,
            num_layers=num_layers,
            neurons_per_layer=neurons_per_layer,
            channels_per_layer=channels_per_layer,
            batch_size=batch_size,
            num_epochs=num_epochs,
            sigma_v=sigma_v,
            learning_rate=learning_rate
        )
        print(f"Finished experiment {idx}/{total_runs}")

if __name__ == "__main__":
    main()