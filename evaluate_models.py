import os
import torch
from src.dataloader import (
    load_data,
    NoisyDataLoader,
)
from src.models_multimodal import (
    load_model,
)
from src.utils import (
    get_valid_dir,
    set_seed,
    get_linear_predictions,
    get_knn_predictions,
    get_embs,
    is_subset,
    process_data_loader,
    print_metrics_in_latex,
    calculate_metrics,
    get_checkpoint_paths,
    mergekfold_results,
    save_normalized_conf_matrices,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load models
set_seed(0)

directories = [
    "models/classification-lc",
    "models/unimodal_lc",
    "models/pretrained_sim_lc_finetuned_lc",
    "models/clip-real-5fold",
    "models/clip-simpretrain-clipreal-5fold",
]  # "ENDtoEND",
names = [
    "class lc",
    "regres lc",
    "masked-sim",
    "clip-real",
    "clip-simpretrain-clipreal",
]
models = []

paths = []
ids = []
labels = []
# Finding all checkpoints
for id, (directory, label) in enumerate(zip(directories, names)):
    paths_to_checkpoint, name, id = get_checkpoint_paths(directory, label, id)
    paths.extend(paths_to_checkpoint)
    ids.extend(id)
    labels.extend(name)


for i, path in enumerate(paths):
    print(f"loading {labels[i]}")
    models.append(load_model(path))

print("finished loading models")


# Data preprocessing

data_dirs = [
    "/home/thelfer1/scr4_tedwar42/thelfer1/ZTFBTS/",
    "ZTFBTS/",
    "/ocean/projects/phy230064p/shared/ZTFBTS/",
    "data/ZTFBTS/",
]
data_dir = get_valid_dir(data_dirs)

data_dirs = [
    "ZTFBTS_spectra/",
    "data/ZTFBTS_spectra/",
    # "/n/home02/gemzhang/Storage/multimodal/ZTFBTS_spectra/",
    # "/n/home02/gemzhang/Storage/multimodal/ZTFBTS_spectra/",
]
spectra_dir = get_valid_dir(data_dirs)


# Default to 1 if the environment variable is not set
cpus_per_task = int(os.getenv("SLURM_CPUS_PER_TASK", 1))

# Assuming you want to leave one CPU for overhead
num_workers = max(1, cpus_per_task - 1)
print(f"Using {num_workers} workers for data loading", flush=True)

# Keeping track of all metrics
regression_metrics_list = []
classification_metrics_list = []
collect_classification_results = []
collect_regression_results = []


for output, label, id in zip(models, labels, ids):
    (
        model,
        combinations,
        regression,
        classification,
        n_classes,
        cfg,
        cfg_extra_args,
        train_filenames,
        val_filenames,
    ) = output

    set_seed(cfg["seed"])

    # Spectral data is cut to this length
    dataset_train, nband, filenames_read, _ = load_data(
        data_dir,
        spectra_dir,
        max_data_len_spec=cfg_extra_args["max_spectral_data_len"],
        combinations=cfg_extra_args["combinations"],
        spectral_rescalefactor=cfg_extra_args["spectral_rescalefactor"],
        filenames=train_filenames,
    )

    # Check that the filenames read are a subset of the training filenames from the already trained models
    assert is_subset(filenames_read, train_filenames)

    dataset_val, nband, filenames_read, _ = load_data(
        data_dir,
        spectra_dir,
        max_data_len_spec=cfg_extra_args["max_spectral_data_len"],
        combinations=cfg_extra_args["combinations"],
        spectral_rescalefactor=cfg_extra_args["spectral_rescalefactor"],
        filenames=val_filenames,
    )

    # Check that the filenames read are a subset of the training filenames from the already trained models
    assert is_subset(filenames_read, val_filenames)

    # val_fraction = cfg_extra_args.get("val_fraction", cfg_extra_args["val_fraction"])
    ## Iterate over data
    # number_of_samples = len(dataset)
    # n_samples_val = int(val_fraction * number_of_samples)

    train_loader_no_aug = NoisyDataLoader(
        dataset_train,
        batch_size=cfg["batchsize"],
        noise_level_img=0,
        noise_level_mag=0,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        combinations=cfg_extra_args["combinations"],
    )

    val_loader_no_aug = NoisyDataLoader(
        dataset_val,
        batch_size=cfg["batchsize"],
        noise_level_img=0,
        noise_level_mag=0,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        combinations=cfg_extra_args["combinations"],
    )

    model = model.to(device)
    model.eval()

    y_true, y_true_label, y_pred = process_data_loader(
        val_loader_no_aug,
        regression,
        classification,
        device,
        model,
        combinations=cfg_extra_args["combinations"],
    )
    y_true_train, y_true_train_label, _ = process_data_loader(
        train_loader_no_aug,
        regression,
        classification,
        device,
        model,
        combinations=cfg_extra_args["combinations"],
    )

    print("===============================")
    print(f"Model: {label}")
    print(f"Using data modalities: {cfg_extra_args['combinations']}")

    def format_combinations(combinations):
        if len(combinations) > 1:
            return ", ".join(combinations[:-1]) + " and " + combinations[-1]
        elif combinations:
            return combinations[0]
        return ""

    if regression:
        metrics, results = calculate_metrics(
            y_true,
            y_true_label,
            y_pred,
            label,
            format_combinations(cfg_extra_args["combinations"]),
            id=id,
            task="regression",
        )
        collect_regression_results.append(results)
        regression_metrics_list.append(metrics)

    elif classification:
        metrics, results = calculate_metrics(
            y_true,
            y_true_label,
            y_pred,
            label,
            format_combinations(cfg_extra_args["combinations"]),
            id=id,
            task="classification",
        )
        classification_metrics_list.append(metrics)
        collect_classification_results.append(results)
    else:
        embs_list, combs = get_embs(
            model, val_loader_no_aug, cfg_extra_args["combinations"], ret_combs=True
        )
        embs_list_train = get_embs(model, train_loader_no_aug, combinations)
        for i in range(len(embs_list)):
            # print(f"Train set linear regression R2 value for {combs[i]}: {get_linearR2(embs_list_train[i], y_true_train)}")
            print(f"---- {combs[i]} input ---- ")
            for task in ["regression", "classification"]:
                if task == "regression":
                    y_pred_linear = get_linear_predictions(
                        embs_list_train[i],
                        y_true_train,
                        embs_list[i],
                        y_true,
                        task=task,
                    )
                    y_pred_knn = get_knn_predictions(
                        embs_list_train[i],
                        y_true_train,
                        embs_list[i],
                        y_true,
                        task=task,
                    )
                    metrics, results = calculate_metrics(
                        y_true,
                        y_true_label,
                        y_pred_linear,
                        label + "+Linear",
                        combs[i],
                        id=id,
                        task=task,
                    )
                    regression_metrics_list.append(metrics)
                    collect_regression_results.append(results)

                    metrics, results = calculate_metrics(
                        y_true,
                        y_true_label,
                        y_pred_knn,
                        label + "+KNN",
                        combs[i],
                        id=id,
                        task=task,
                    )
                    regression_metrics_list.append(metrics)
                    collect_regression_results.append(results)

                elif task == "classification":
                    y_pred_linear = get_linear_predictions(
                        embs_list_train[i],
                        y_true_train_label,
                        embs_list[i],
                        y_true_label,
                        task=task,
                    )
                    y_pred_knn = get_knn_predictions(
                        embs_list_train[i],
                        y_true_train_label,
                        embs_list[i],
                        y_true_label,
                        task=task,
                    )
                    metrics, results = calculate_metrics(
                        y_true,
                        y_true_label,
                        y_pred_linear,
                        label + "+Linear",
                        combs[i],
                        id=id,
                        task=task,
                    )
                    classification_metrics_list.append(metrics)
                    collect_classification_results.append(results)

                    metrics, results = calculate_metrics(
                        y_true,
                        y_true_label,
                        y_pred_knn,
                        label + "+KNN",
                        combs[i],
                        id=id,
                        task=task,
                    )
                    collect_classification_results.append(results)
                    classification_metrics_list.append(metrics)

        # for concatenated pairs of modalities
        for i in range(len(embs_list)):
            for j in range(i + 1, len(embs_list)):
                emb_concat = torch.cat([embs_list[i], embs_list[j]], dim=1)
                emb_train = torch.cat([embs_list_train[i], embs_list_train[j]], dim=1)
                print(f"---- {combs[i]} and {combs[j]} input ---- ")
                for task in ["regression", "classification"]:
                    if task == "regression":
                        y_pred_linear = get_linear_predictions(
                            embs_list_train[i],
                            y_true_train,
                            embs_list[i],
                            y_true,
                            task=task,
                        )
                        y_pred_knn = get_knn_predictions(
                            embs_list_train[i],
                            y_true_train,
                            embs_list[i],
                            y_true,
                            task=task,
                        )
                        metrics, results = calculate_metrics(
                            y_true,
                            y_true_label,
                            y_pred_linear,
                            label + "+Linear",
                            combs[i] + " and " + combs[j],
                            id=id,
                            task=task,
                        )
                        regression_metrics_list.append(metrics)
                        collect_regression_results.append(results)

                        metrics, results = calculate_metrics(
                            y_true,
                            y_true_label,
                            y_pred_knn,
                            label + "+KNN",
                            combs[i] + " and " + combs[j],
                            id=id,
                            task=task,
                        )
                        collect_regression_results.append(results)
                        regression_metrics_list.append(metrics)
                    elif task == "classification":
                        y_pred_linear = get_linear_predictions(
                            embs_list_train[i],
                            y_true_train_label,
                            embs_list[i],
                            y_true_label,
                            task=task,
                        )
                        y_pred_knn = get_knn_predictions(
                            embs_list_train[i],
                            y_true_train_label,
                            embs_list[i],
                            y_true_label,
                            task=task,
                        )
                        metrics, results = calculate_metrics(
                            y_true,
                            y_true_label,
                            y_pred_linear,
                            label + "+Linear",
                            combs[i] + " and " + combs[j],
                            id=id,
                            task=task,
                        )
                        classification_metrics_list.append(metrics)
                        collect_classification_results.append(results)

                        metrics, results = calculate_metrics(
                            y_true,
                            y_true_label,
                            y_pred_knn,
                            label + "+KNN",
                            combs[i] + " and " + combs[j],
                            id=id,
                            task=task,
                        )
                        collect_classification_results.append(results)
                        classification_metrics_list.append(metrics)
    print("===============================")

# Convert metrics list to a DataFrame
print_metrics_in_latex(classification_metrics_list)
print_metrics_in_latex(regression_metrics_list)
if len(collect_classification_results) > 0:
    merged_classification = mergekfold_results(collect_classification_results)
    save_normalized_conf_matrices(merged_classification, "confusion_plots")

if len(regression_metrics_list) > 0:
    merged_regression = mergekfold_results(regression_metrics_list)
