import ast
import re
import json
from os import makedirs, path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity


# =============================================================================
# 1. DATA PARSING & UTILITIES
# =============================================================================

def parse_score_dictionary(val) -> dict:
    """Safely parses dictionary input whether provided as string, dict, or stringified dict."""
    if pd.isna(val) or val is None:
        return {}
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        val = val.strip()
        # Try evaluating as literal Python dictionary
        if val.startswith("{") and val.endswith("}"):
            try:
                return ast.literal_eval(val)
            except Exception:
                pass
        # Fallback parse comma-separated format: "attribute (0.50), risk-averse (1.0)"
        res = {}
        for item in val.split(","):
            match = re.match(r'^(.*?)\s*\(([-+]?\d*\.?\d+)\)$', item.strip())
            if match:
                res[match.group(1).strip()] = float(match.group(2))
        return res
    return {}


# =============================================================================
# 2. TRAIT SPARSITY & DENSITY ANALYSIS
# =============================================================================

def analyze_nonzero_trait_counts(df: pd.DataFrame, eval_cols: dict, threshold: float = 0.0) -> pd.DataFrame:
    """Calculates summary statistics on the number of active (non-zero) traits

    assigned per document across evaluators.
    """
    summary_records = []
    for label, col in eval_cols.items():
        counts = df[col].apply(
            lambda d: sum(1 for v in d.values() if abs(v) > threshold) if isinstance(d, dict) else 0
        )
        summary_records.append({
            "Evaluator": label,
            "Mean_Active_Traits": round(counts.mean(), 2),
            "Std_Dev": round(counts.std(), 2),
            "Min": counts.min(),
            "Median": counts.median(),
            "Max": counts.max()
        })
    return pd.DataFrame(summary_records)


def compute_teacher_coverage_alignment(
    df: pd.DataFrame, 
    ref_col: str = "openai_evaluation", 
    target_cols: dict = None, 
    threshold: float = 0.0
) -> dict:
    """Measures what proportion of the Teacher's active (non-zero) traits

    were also identified by other evaluators on a per-document basis.
    """
    if target_cols is None:
        target_cols = {"Student": "original_evaluation", "Human": "human_evaluation"}

    alignment_results = {}
    for label, col in target_cols.items():
        overlap_ratios = []
        for item in df.itertuples(index=False):
            teacher_dict = getattr(item, ref_col, {}) or {}
            eval_dict = getattr(item, col, {}) or {}

            teacher_active = {k for k, v in teacher_dict.items() if abs(v) > threshold}
            eval_active = {k for k, v in eval_dict.items() if abs(v) > threshold}

            if len(teacher_active) > 0:
                overlap = len(teacher_active.intersection(eval_active)) / len(teacher_active)
                overlap_ratios.append(overlap)

        alignment_results[label] = round(np.mean(overlap_ratios), 3) if overlap_ratios else 0.0

    return alignment_results


# =============================================================================
# 3. ROW-LEVEL VECTOR SIMILARITY ANALYSIS vs TEACHER
# =============================================================================

def compute_metrics_vs_teacher(
    df: pd.DataFrame, 
    col_ref: str = "openai_evaluation", 
    col_eval: str = "human_evaluation"
):
    """Computes document-level Pearson correlation, Cosine similarity, and MSE

    taking col_ref (Teacher) as the ground truth reference.
    """
    corr, cosine_sim, mse = [], [], []

    for item in df.itertuples(index=False):
        ref_val = getattr(item, col_ref, None)
        eval_val = getattr(item, col_eval, None)

        if not isinstance(ref_val, dict) or not isinstance(eval_val, dict):
            continue

        s_ref = pd.Series(ref_val, name="teacher_ref")
        s_eval = pd.Series(eval_val, name="evaluator")
        comp_df = pd.concat([s_ref, s_eval], axis=1).fillna(0.0)

        # Pearson Correlation
        if len(comp_df) > 1 and comp_df["teacher_ref"].nunique() > 1 and comp_df["evaluator"].nunique() > 1:
            r = comp_df.corr(method="pearson").loc["teacher_ref", "evaluator"]
            if not pd.isna(r):
                corr.append(r)

        # Cosine Similarity
        vec_ref = comp_df["teacher_ref"].values.reshape(1, -1)
        vec_eval = comp_df["evaluator"].values.reshape(1, -1)
        if np.any(vec_ref) or np.any(vec_eval):
            cos_score = cosine_similarity(vec_ref, vec_eval)[0][0]
            if not pd.isna(cos_score):
                cosine_sim.append(cos_score)

        # Mean Squared Error against Teacher Reference
        mse_score = mean_squared_error(comp_df["teacher_ref"], comp_df["evaluator"])
        mse.append(mse_score)

    return corr, cosine_sim, mse


# =============================================================================
# 4. DIAGNOSTIC PLOTS
# =============================================================================

def plot_density_comparison(sparsity_df: pd.DataFrame, output_folder: str):
    """Generates a bar chart comparing active non-zero trait counts across evaluators."""
    plt.figure(figsize=(8, 5))
    sns.set_theme(style="whitegrid")
    sns.barplot(data=sparsity_df, x="Evaluator", y="Mean_Active_Traits", hue="Evaluator", legend=False, palette="Blues_d")
    plt.ylabel("Mean Active (Non-Zero) Traits per Document")
    plt.tight_layout()
    plt.savefig(path.join(output_folder, "plot_trait_density_comparison.pdf"), dpi=300)
    plt.close()


# =============================================================================
# 5. MAIN PIPELINE ORCHESTRATION
# =============================================================================

def print_metric_summary(metric_name: str, values: list):
    """Prints descriptive statistics for calculated metrics."""
    print(f"\n{metric_name} Summary:")
    if values:
        print(pd.Series(values).describe().round(2).to_frame().T)
    else:
        print("No valid scores computed.")

def compute_attribute_wise_differences(
    df: pd.DataFrame, 
    grouped_traits: dict,
    col_ref: str, 
    col_eval: str, 
    fill_value: float = 0.0
) -> pd.DataFrame:
    """Computes attribute-level metrics (Mean Ref, Mean Eval, Bias, MAE, RMSE) 

    between a reference evaluator and a target evaluator.
    """
    # Expand nested dictionaries into DataFrames
    ref_df = pd.DataFrame(df[col_ref].tolist(), index=df.index)
    eval_df = pd.DataFrame(df[col_eval].tolist(), index=df.index)

    # Union all unique attributes and align DataFrames
    all_attrs = sorted(set(ref_df.columns) | set(eval_df.columns))
    ref_df = ref_df.reindex(columns=all_attrs).fillna(fill_value)
    eval_df = eval_df.reindex(columns=all_attrs).fillna(fill_value)

    # Vectorized error calculations across all attributes
    diff = eval_df - ref_df
    abs_diff = np.abs(diff)

    ref_label = col_ref.replace('_evaluation', '')
    eval_label = col_eval.replace('_evaluation', '')

    summary_df = pd.DataFrame({
        "Attribute": all_attrs,
        "Domain": [grouped_traits.get(att).replace('_', ' ').title() for att in all_attrs],
        f"{ref_label} (Mean)": ref_df.mean().values,
        f"{eval_label} (Mean)": eval_df.mean().values,
        "diff": diff.mean().values,
        "MAE": abs_diff.mean().values,
        "RMSE": np.sqrt((diff ** 2).mean().values)
    })


    return summary_df.sort_values(by="MAE", ascending=False).reset_index(drop=True)


def main():
    output_folder = path.join("outputs")
    makedirs(output_folder, exist_ok=True)

    # get the domain
    with open(path.join('data', 'selected_traits.json')) as jf:
        grouped_traits = json.load(jf)
    grouped_traits = {item: group for group, items in grouped_traits.items() for item in items}
    
    # Load Human Evaluations
    human_evals = pd.read_csv(path.join('data', "human_evaluations_rows.csv"))
    if 'scores' in human_evals.columns:
        human_evals['human_evaluation'] = human_evals['scores'].apply(parse_score_dictionary)
    else:
        human_evals['human_evaluation'] = human_evals['human_evaluation'].apply(parse_score_dictionary)
    
    # Load LLM Evaluations
    llm_evals = pd.read_json(path.join('data', "text_openai_eval.jsonl"), lines=True).reset_index(names="doc_id")
    llm_evals['openai_evaluation'] = llm_evals['openai_evaluation'].apply(parse_score_dictionary)
    llm_evals['original_evaluation'] = llm_evals['original_evaluation'].apply(parse_score_dictionary)
    
    llm_evals['original_evaluation'] = llm_evals['original_evaluation'].apply(
        lambda x: {att: round(score, 1) for att, score in x.items()}
    )

    # Merge Data
    df = pd.merge(human_evals, llm_evals, how="inner", on="doc_id")

    # 1. Annotation Density & Sparsity
    print("\n" + "=" * 80)
    print("1. ANNOTATION DENSITY & SPARSITY ANALYSIS")
    print("=" * 80)

    eval_cols = {
        "Teacher (OpenAI)": "openai_evaluation",
        "Student (Original)": "original_evaluation",
        "Human Annotators": "human_evaluation"
    }

    sparsity_df = analyze_nonzero_trait_counts(df, eval_cols)
    print(sparsity_df.to_string(index=False))

    plot_density_comparison(sparsity_df, output_folder)

    coverage = compute_teacher_coverage_alignment(
        df, ref_col="openai_evaluation",
        target_cols={"Student": "original_evaluation", "Human": "human_evaluation"}
    )
    print("\n--- Teacher Trait Coverage / Active Recall ---")
    for k, v in coverage.items():
        print(f"  * {k} captures {v * 100:.1f}% of Teacher's active non-zero traits on average.")

    # 2. Row-Level Vector Metrics
    print("\n" + "=" * 80)
    print("2. ROW-LEVEL VECTOR SIMILARITY vs TEACHER REFERENCE")
    print("=" * 80)

    comparisons = [
        ("openai_evaluation", "human_evaluation"),
        ("openai_evaluation", "original_evaluation"),
        ("original_evaluation", "human_evaluation"),
        ("original_evaluation", "openai_evaluation"),
    ]

    for id_comp, (ref_col, eval_col) in enumerate(comparisons):
        ref_label, eval_label = ref_col.replace('_evaluation', ''), eval_col.replace('_evaluation', '')

        print(f"\n{'='*50}\n({id_comp + 1}) {ref_label} vs {eval_label}\n{'='*50}")

        corr, cosine_sim, mse = compute_metrics_vs_teacher(
            df, col_ref=ref_col, col_eval=eval_col
        )
        print_metric_summary("Pearson Correlation", corr)
        print_metric_summary("Cosine Similarity", cosine_sim)
        print_metric_summary(f"MSE (vs {ref_label})", mse)

        # 2. Attribute-wise Differences
        attr_diffs = compute_attribute_wise_differences(df, grouped_traits, col_ref=ref_col, col_eval=eval_col)

        with pd.ExcelWriter(path.join(output_folder, 'attribute_diffs.xlsx'), mode = 'w' if id_comp == 0 else "a") as xf:
            attr_diffs.to_excel(xf, sheet_name = f'{ref_label}_{eval_label}')

        if (ref_col == 'original_evaluation') and  (eval_col == 'human_evaluation'):
            attr_diffs.round(2).to_latex(path.join(output_folder, 'dff_original_human.tex'), index = False)


        print(f"\n--- Top 10 Attribute Differences ({eval_label} vs {ref_label}) ---")
        print(attr_diffs.head(10).to_string(index=False))



if __name__ == "__main__":
    main()
