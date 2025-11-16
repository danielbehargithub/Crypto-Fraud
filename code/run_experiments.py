import copy
import pandas as pd
from data import get_variants
from active_learning import run_active_learning
from training import run
import yaml

ECFG = yaml.safe_load(open("configs/config_run_experiments.yaml"))

def run_experiments(
    run_type: str,
    *,
    data_variants,
    graph_modes,
    model_names,
    feature_sets,
    split_types,
    al_methods = None,
):
    """
    Run a grid of experiments over models / graph modes / feature sets / splits.

    Parameters
    ----------
    run_type : Either "passive" or "active".
        - "passive": standard supervised training using `training.run`.
        - "active": pool-based Active Learning using `run_active_learning`.
    data_variants : A dictionary mapping (graph_mode, features_set, split_type) tuples
        to preconstructed `Data` objects, typically created by `get_variants`.
    graph_modes : Graph construction modes, e.g. ["dag"], ["dag", "undirected"].
    model_names : List of model identifiers to evaluate, e.g. ["MLP", "GCN"].
    feature_sets : Feature configurations, e.g. ["local"], ["local", "all"].
    split_types : Train/val/test split strategies, e.g. ["temporal"], ["temporal", "random"].
    al_methods : List of acquisition strategies for Active Learning runs
        (e.g. ["entropy", "random", "umcs", "sequential"]).
        Only used when run_type == "active".

    Returns
    -------
    pandas.DataFrame
        A table where each row summarizes a single experiment configuration,
        including model name, data variant tags, and evaluation metrics.
    """
    al_cfg = ECFG["al_params"]
    rows = []
    if run_type == "passive":
        for model_name in model_names:
            for gm in graph_modes:
                for fset in feature_sets:
                    for split in split_types:
                        data_obj = data_variants[(gm, fset, split)]
                        row = run(data_obj, model_name, fset, split, gm)
                        rows.append(row)
    elif run_type == "active":
        model_variants_cfg = ECFG.get("model_variants", {})
        for model_name in model_names:
            mv = model_variants_cfg.get(model_name.upper(), {})
            mv = mv[0]
            chosen_gm   = mv["graph_mode"]
            chosen_fset = mv["features_set"]
            chosen_split = mv["split_type"]
            data_obj = data_variants[(chosen_gm, chosen_fset, chosen_split)]
            for method in al_methods:
                data_copy = copy.deepcopy(data_obj)
                row = run_active_learning(
                    data=data_copy,
                    model_name=model_name,
                    features_set=chosen_fset,
                    split_type=chosen_split,
                    graph_mode=chosen_gm,
                    seed_per_class=al_cfg["seed_per_class"],
                    batch_size=al_cfg["batch_size"],
                    budget=al_cfg["budget"],
                    max_epochs_per_round=al_cfg["max_epochs_per_round"],
                    rng_seed=al_cfg["rng_seed"],
                    method=method,
                )
                rows.append(row)
    return pd.DataFrame(rows)


def main():
    """
    Entry point for running all configured experiments.

    The function:
      1. Reads the experiment configuration from `config_run_experiments.yaml`
         (already loaded into ECFG).
      2. Builds the requested data variants via `get_variants`, for all
         combinations of graph_modes, feature_sets and split_types.
      3. Runs:
         - Passive experiments (standard supervised training) and writes
           a summary CSV file with their metrics.
         - Active Learning experiments for each acquisition method and
           writes both a summary CSV and a per-round curves CSV.
      4. Prints human-readable summary tables to the console.

    This file can be used both as a script (via `python run_experiments.py`)
    and as an importable module where `run_experiments`/`main` are reused.
    """

    # Choose subsets here:
    graph_modes = ECFG["graph_modes"]
    model_names = ECFG["model_names"]
    feature_sets = ECFG["feature_sets"]
    split_types = ECFG["split_types"]
    al_methods = ECFG["al_methods"]

    # Build only the variants you need (saves memory/time)
    data_variants = get_variants(
        graph_modes=tuple(graph_modes),
        feature_sets=tuple(feature_sets),
        split_types=tuple(split_types),
    )

    #---- Passive runs ----
    if ECFG.get("run_passive", True):
        df_passive = run_experiments(
            "passive",
            data_variants=data_variants,
            graph_modes=graph_modes,
            model_names=model_names,
            feature_sets=feature_sets,
            split_types=split_types,
        )
        if not df_passive.empty:
            print("\n=== Summary Table (Passive runs) ===")
            df_passive = df_passive.sort_values(by="test_f1", ascending=False)
            print(df_passive.to_string(index=False))
            passive_csv = ECFG["output"]["passive_summary_csv"]
            df_passive.to_csv(passive_csv, index=False)

    # ---- Active Learning runs ----
    if ECFG.get("run_active", True):
        df_active = run_experiments(
            "active",
            data_variants=data_variants,
            graph_modes=graph_modes,
            model_names=model_names,
            feature_sets=feature_sets,
            split_types=split_types,
            al_methods=al_methods,
        )
        if not df_active.empty:
            cols_no_curve = [c for c in df_active.columns if c != "curve"]
            df_active_summary = df_active[cols_no_curve].copy()

            df_active_summary = df_active_summary.sort_values(by="test_f1", ascending=False)
            active_csv = ECFG["output"]["active_summary_csv"]
            df_active_summary.to_csv(active_csv, index=False)
            print("\n=== Summary Table (Active Learning runs) ===")
            print(df_active_summary.to_string(index=False))

            df_curves = (
                df_active[["method", "model", "features_set", "split_type", "graph_mode", "in_channels", "curve"]]
                .explode("curve", ignore_index=True)
            )
            curve_flat = pd.json_normalize(df_curves["curve"])
            df_curves = pd.concat([df_curves.drop(columns=["curve"]), curve_flat], axis=1)
            curves_csv = ECFG["output"]["active_curves_csv"]
            df_curves.to_csv(curves_csv, index=False)


if __name__ == "__main__":
    main()
