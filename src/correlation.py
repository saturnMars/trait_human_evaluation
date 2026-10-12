from os import path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity

def compute_metrics_for_pair(df: pd.DataFrame, col_a: str, col_b: str):
    """Computes row-level Pearson correlation, Cosine similarity, and MSE

    between score dictionaries/lists across two DataFrame columns.
    """
    corr, cosine_sim, mse = [], [], []

    for item in df.itertuples(index=False):
        val_a = getattr(item, col_a, None)
        val_b = getattr(item, col_b, None)

        if val_a is None or val_b is None or pd.isna(val_a) or pd.isna(val_b):
            continue

        # Convert dictionary/list evaluations into aligned Series
        s_a = pd.Series(val_a, name=col_a)
        s_b = pd.Series(val_b, name=col_b)
        comp_df = pd.concat([s_a, s_b], axis=1).fillna(0.0)

        # Pearson Correlation
        if len(comp_df) > 1 and comp_df[col_a].nunique() > 1 and comp_df[col_b].nunique() > 1:
            r = comp_df.corr(method="pearson").loc[col_a, col_b]
            if not pd.isna(r):
                corr.append(r)

        # Cosine Similarity
        vec_a = comp_df[col_a].values.reshape(1, -1)
        vec_b = comp_df[col_b].values.reshape(1, -1)
        if np.any(vec_a) or np.any(vec_b):
            cos_score = cosine_similarity(vec_a, vec_b)[0][0]
            if not pd.isna(cos_score):
                cosine_sim.append(cos_score)

        # Mean Squared Error
        mse_score = mean_squared_error(comp_df[col_a], comp_df[col_b])
        mse.append(mse_score)

    return corr, cosine_sim, mse


def print_metric_summary(metric_name: str, values: list):
    """Prints descriptive statistics for calculated metrics."""
    print(f"\n{metric_name} Summary:")
    if values:
        print(pd.Series(values).describe().round(2))
    else:
        print("No valid scores computed.")


def main():
    output_folder = path.join("outputs")

    # Load human evaluations
    human_evals = pd.read_json(path.join(output_folder, "human_evaluations.jsonl"), lines=True)
    human_evals = human_evals.rename(columns={"scores": "human_evaluation"})

    # Load LLM-based evaluations
    llm_evals = pd.read_json(path.join('data', "text_openai_eval.jsonl"), lines=True).reset_index(names="doc_id")

    # Merge evaluation DataFrames
    df = pd.merge(llm_evals, human_evals, how="inner", on="doc_id")

    # Metric evaluation pairs
    pairs = [
        ("openai_evaluation", "human_evaluation"),
        ("original_evaluation", "human_evaluation"),
    ]

    for col_a, col_b in pairs:
        print(f"\n{'='*50}\nEvaluating Pair: {col_a} vs {col_b}\n{'='*50}")
        corr, cosine_sim, mse = compute_metrics_for_pair(df, col_a, col_b)

        print_metric_summary("Pearson Correlation", corr)
        print_metric_summary("Cosine Similarity", cosine_sim)
        print_metric_summary("MSE", mse)


if __name__ == "__main__":
    main()