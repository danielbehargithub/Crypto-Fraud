import copy
import pandas as pd
from data import get_variants
from active_learning import run_active_learning
from training import run


def run_experiments(
    run_type: str,
    *,
    data_variants,
    graph_modes,
    model_names,
    feature_sets,
    split_types,
    al_acquisitions = None,
):
    """
    Generic experiment loop. Uses configs provided by main().
    run_type: "passive" or "active".
    """
    rows = []
    for gm in graph_modes:
        for model_name in model_names:
            for fset in feature_sets:
                for split in split_types:
                    data_obj = data_variants[(gm, fset, split)]

                    if run_type == "passive":
                        row = run(data_obj, model_name, fset, split, gm)
                        rows.append(row)
                    elif run_type == "active":
                        for acquisition in al_acquisitions:
                            data_copy = copy.deepcopy(data_obj)
                            row = run_active_learning(
                                data=data_copy,
                                model_name=model_name,
                                features_set=fset,
                                split_type=split,
                                graph_mode=gm,
                                seed_per_class=10,
                                batch_size=20,
                                budget=20,
                                max_epochs_per_round=20,
                                rng_seed=42,
                                acquisition=acquisition,
                            )
                            rows.append(row)
    return pd.DataFrame(rows)


def main():
    # Choose subsets here:
    graph_modes  = ["dag", "undirected"]   # or ["dag"] or ["undirected"]
    model_names  = ["MLP", "GCN"]                  # e.g., ["GCN"] to run only GCN. MLP available
    feature_sets = ["local", "all"]                # e.g., ["local"] to run only LOCAL. all available
    split_types  = ["temporal"]             # e.g., ["temporal"] to run only TEMPORAL. random available

    al_acquisitions = ["entropy", "random", "umcs", "sequential"]  # choose between "entropy", "random", or both: ["entropy", "random"]

    # Build only the variants you need (saves memory/time)
    data_variants = get_variants(
        graph_modes=tuple(graph_modes),
        feature_sets=tuple(feature_sets),
        split_types=tuple(split_types),
    )

    #---- Passive runs ----
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
        df_passive.to_csv("results/run_summary_passive.csv", index=False)

    # ---- Active Learning runs ----
    df_active = run_experiments(
        "active",
        data_variants=data_variants,
        graph_modes=graph_modes,
        model_names=model_names,
        feature_sets=feature_sets,
        split_types=split_types,
        al_acquisitions=al_acquisitions,
    )
    if not df_active.empty:
        cols_no_curve = [c for c in df_active.columns if c != "curve"]
        df_active_summary = df_active[cols_no_curve].copy()

        print("\n=== Summary Table (Active Learning runs) ===")
        df_active_summary = df_active_summary.sort_values(by="test_f1", ascending=False)
        print(df_active_summary.to_string(index=False))
        df_active_summary.to_csv("results/run_summary_active.csv", index=False)
        df_curves = (
            df_active[["acquisition", "model", "features_set", "split_type", "graph_mode", "in_channels", "curve"]]
            .explode("curve", ignore_index=True)
        )
        curve_flat = pd.json_normalize(df_curves["curve"])
        df_curves = pd.concat([df_curves.drop(columns=["curve"]), curve_flat], axis=1)
        df_curves.to_csv("results/run_curves.csv", index=False)
if __name__ == "__main__":
    main()
