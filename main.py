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
    al_acquisitions = ("entropy",),
):
    """
    Generic experiment loop. Uses configs provided by main().
    run_type: "passive" or "active".
    """

    if run_type == "active":
        if isinstance(al_acquisitions, str):
            acquisitions_iter = (al_acquisitions,)
        else:
            acquisitions_iter = tuple(al_acquisitions)
        if len(acquisitions_iter) == 0:
            raise ValueError("al_acquisitions must contain at least one strategy for active runs.")
    else:
        acquisitions_iter = ()

    rows = []
    for gm in graph_modes:
        for model_name in model_names:
            for fset in feature_sets:
                for split in split_types:
                    data_obj = data_variants[(gm, fset, split)]

                    if run_type == "passive":
                        row = run(data_obj, model_name, fset, split, gm)
                    elif run_type == "active":
                        for acquisition in al_acquisitions:
                            row = run_active_learning(
                                data=data_obj,
                                model_name=model_name,
                                features_set=fset,
                                split_type=split,
                                graph_mode=gm,
                                seed_per_class=10,
                                batch_size=50,
                                budget=200,
                                max_epochs_per_round=100,
                                rng_seed=42,
                                acquisition=acquisition,
                            )
                    else:
                        raise ValueError(f"Unknown run_type: {run_type}")

                    rows.append(row)

    return pd.DataFrame(rows)


def main():
    # Choose subsets here:
    graph_modes  = ["dag"]   # or ["dag"] or ["undirected"]
    model_names  = ["MLP"]                  # e.g., ["GCN"] to run only GCN. MLP available
    feature_sets = ["local"]                # e.g., ["local"] to run only LOCAL. all available
    split_types  = ["temporal"]             # e.g., ["temporal"] to run only TEMPORAL. random available

    al_acquisition = "random"  # choose between "entropy" and "random"

    # Build only the variants you need (saves memory/time)
    data_variants = get_variants(
        graph_modes=tuple(graph_modes),
        feature_sets=tuple(feature_sets),
        split_types=tuple(split_types),
    )

    # ---- Passive runs ----
    # df_passive = run_experiments(
    #     "passive",
    #     data_variants=data_variants,
    #     graph_modes=graph_modes,
    #     model_names=model_names,
    #     feature_sets=feature_sets,
    #     split_types=split_types,
    # )
    # if not df_passive.empty:
    #     print("\n=== Summary Table (Passive runs) ===")
    #     print(df_passive.to_string(index=False))
    #     df_passive.to_csv("run_summary_passive.csv", index=False)

    # ---- Active Learning runs ----
    df_active = run_experiments(
        "active",
        data_variants=data_variants,
        graph_modes=graph_modes,
        model_names=model_names,
        feature_sets=feature_sets,
        split_types=split_types,
        al_acquisition=al_acquisition,
    )
    if not df_active.empty:
        print("\n=== Summary Table (Active Learning runs) ===")
        print(df_active.to_string(index=False))
        df_active.to_csv("run_summary_active.csv", index=False)


if __name__ == "__main__":
    main()
