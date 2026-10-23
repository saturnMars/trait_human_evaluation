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
# 4. POLARITY-BASED ERROR ANALYSIS & ATTRIBUTE DIFFERENCES
# =============================================================================

def analyze_errors_by_score_polarity(
    df: pd.DataFrame, 
    col_ref: str, 
    col_eval: str, 
    fill_value: float = 0.0
) -> pd.DataFrame:
    """Groups evaluations into positive (>0), negative (<0), and zero (=0) reference 
    score buckets and computes error metrics (MAE, RMSE, Bias, Sign Agreement, Sign Flips).
    """
    ref_df = pd.DataFrame(df[col_ref].tolist(), index=df.index).fillna(fill_value)
    eval_df = pd.DataFrame(df[col_eval].tolist(), index=df.index).fillna(fill_value)

    all_attrs = sorted(set(ref_df.columns) | set(eval_df.columns))
    ref_df = ref_df.reindex(columns=all_attrs).fillna(fill_value)
    eval_df = eval_df.reindex(columns=all_attrs).fillna(fill_value)

    ref_flat = ref_df.values.flatten()
    eval_flat = eval_df.values.flatten()
    diff_flat = eval_flat - ref_flat

    pos_mask = ref_flat > 0
    neg_mask = ref_flat < 0
    zero_mask = ref_flat == 0

    categories = [
        ("Positive Scores", pos_mask),
        ("Negative Scores", neg_mask),
        ("Zero Scores", zero_mask),
    ]

    records = []
    for label, mask in categories:
        if not np.any(mask):
            continue

        r_vals = ref_flat[mask]
        e_vals = eval_flat[mask]
        d_vals = diff_flat[mask]
        abs_d = np.abs(d_vals)

        print('d_vals', d_vals)

        exit()

        # Sign agreement: exact same sign direction
        sign_match = np.sign(r_vals) == np.sign(e_vals)

        # Sign flip: non-zero opposite signs (e.g., ref > 0 and eval < 0)
        sign_flip = (r_vals * e_vals) < 0

        ref_label = col_ref.replace('_evaluation', '').replace('original', 'OUR')
        eval_label = col_eval.replace('_evaluation', '').replace('original', 'OUR')

        records.append({
            "Polarity Group": label,
            "Count": int(np.sum(mask)),
            f"Mean Score ({ref_label})": round(float(np.mean(r_vals)), 3),
            f"Mean Score ({eval_label})": round(float(np.mean(e_vals)), 3),
            "": round(float(np.mean(d_vals)), 3),
            "MAE": round(float(np.mean(abs_d)), 3),
            "RMSE": round(float(np.sqrt(np.mean(d_vals ** 2))), 3),
            "Sign Agreement (%)": round(float(np.mean(sign_match) * 100), 2),
            "Sign Flip Rate (%)": round(float(np.mean(sign_flip) * 100), 2)
        })

    return pd.DataFrame(records)


def compute_attribute_wise_differences(
    df: pd.DataFrame, 
    grouped_traits: dict,
    col_ref: str, 
    col_eval: str, 
    fill_value: float = 0.0
) -> pd.DataFrame:
    """Computes attribute-level metrics (Mean Ref, Mean Eval, Bias, MAE, RMSE, Polarity)
    between a reference evaluator and a target evaluator.
    """
    ref_df = pd.DataFrame(df[col_ref].tolist(), index=df.index)
    eval_df = pd.DataFrame(df[col_eval].tolist(), index=df.index)

    all_attrs = sorted(set(ref_df.columns) | set(eval_df.columns))
    ref_df = ref_df.reindex(columns=all_attrs).fillna(fill_value)
    eval_df = eval_df.reindex(columns=all_attrs).fillna(fill_value)

    diff = eval_df - ref_df
    abs_diff = np.abs(diff)

    ref_label = col_ref.replace('_evaluation', '')
    eval_label = col_eval.replace('_evaluation', '')

    mean_ref = ref_df.mean().values
    polarities = [
        "Positive" if val > 0.05 else ("Negative" if val < -0.05 else "Neutral") 
        for val in mean_ref
    ]

    summary_df = pd.DataFrame({
        "Attribute": all_attrs,
        "Domain": [grouped_traits.get(att, "Unknown").replace('_', ' ').title() for att in all_attrs],
        "Predominant Polarity": polarities,
        f"{ref_label} (Mean)": mean_ref,
        f"{eval_label} (Mean)": eval_df.mean().values,
        "diff": diff.mean().values,
        "MAE": abs_diff.mean().values,
        "RMSE": np.sqrt((diff ** 2).mean().values)
    })

    return summary_df.sort_values(by="MAE", ascending=False).reset_index(drop=True)


# =============================================================================
# 5. DIAGNOSTIC & ERROR PLOTS
# =============================================================================

def plot_density_comparison(sparsity_df: pd.DataFrame, output_folder: str):
    """Generates a bar chart comparing active non-zero trait counts across evaluators."""
    plt.figure(figsize=(8, 5))
    sns.set_theme(style="whitegrid")
    sns.barplot(data=sparsity_df, x="Evaluator", y="Mean_Active_Traits", hue="Evaluator", legend=False, palette="Blues_d")
    plt.ylabel("Mean Active (Non-Zero) Traits per Document")
    plt.tight_layout()
    plt.savefig(path.join(output_folder, "trait_density_comparison.pdf"), dpi=300)
    plt.close()


def plot_raw_score_distributions(df: pd.DataFrame, eval_cols: dict, output_folder: str):
    """Generates a box plot comparing the distribution of all raw scores across evaluators."""
    records = []
    for label, col in eval_cols.items():
        for score_dict in df[col]:
            if isinstance(score_dict, dict):
                for attr, score in score_dict.items():
                    records.append({
                        "Evaluator": label,
                        "Attribute": attr,
                        "Score": float(score)
                    })

    raw_df = pd.DataFrame(records)
    if raw_df.empty:
        return

    raw_df = raw_df[raw_df['Score'].abs() > 0.05]

    print(raw_df)
    print(raw_df.iloc[0])



    plt.figure(figsize=(9, 6))
    sns.set_theme(style="whitegrid")

    sns.boxplot(
        data=raw_df,
        x="Evaluator",
        y="Score",
        hue="Evaluator",
        palette="Set2",
        legend=False,
    )

    plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    #plt.title("Distribution of All Raw Trait Scores Across Evaluators\n(Red dots indicate mean score)", fontsize=12)
    plt.xlabel("Evaluator")
    plt.ylabel("Non-zero Score")
    plt.tight_layout()

    plt.savefig(path.join(output_folder, "raw_score_distributions.pdf"), dpi=300)
    plt.close()


def plot_attribute_differences(
    summary_df: pd.DataFrame, 
    ref_label: str, 
    eval_label: str, 
    output_folder: str, 
    top_n: int = 15
):
    """Generates a horizontal bar chart visualizing the top positive and negative 
    mean differences across attributes.
    """
    df_sorted = summary_df.sort_values(by="diff", ascending=False)
    top_positive = df_sorted.head(top_n)
    top_negative = df_sorted.tail(top_n)
    plot_df = pd.concat([top_positive, top_negative]).drop_duplicates()

    plt.figure(figsize=(10, 8))
    sns.set_theme(style="whitegrid")
    
    colors = ['#2ca02c' if x >= 0 else '#d62728' for x in plot_df["diff"]]
    
    sns.barplot(
        data=plot_df, 
        y="Attribute", 
        x="diff", 
        palette=colors,
        hue="Attribute",
        legend=False
    )
    plt.axvline(0, color="black", linestyle="--", linewidth=1)
    plt.title(f"Average Trait Score Differences ({eval_label} minus {ref_label})\n(Top Positive & Negative Biases)", fontsize=12)
    plt.xlabel(f"Mean Difference ({eval_label} - {ref_label})")
    plt.ylabel("Trait Attribute")
    plt.tight_layout()
    
    fig_name = f"attr_diff_{ref_label}_vs_{eval_label}.pdf"
    plt.savefig(path.join(output_folder, fig_name), dpi=300)
    plt.close()


def plot_polarity_error_summary(
    polarity_df: pd.DataFrame, 
    ref_label: str, 
    eval_label: str, 
    output_folder: str
):
    """Plots MAE and Mean Bias grouped across Positive, Negative, and Zero reference score buckets."""
    plt.figure(figsize=(8, 5))
    sns.set_theme(style="whitegrid")
    
    df_melted = polarity_df.melt(
        id_vars=["Polarity Group"], 
        value_vars=["MAE", "mean Δ (Bias)"], 
        var_name="Metric", 
        value_name="Value"
    )
    
    sns.barplot(data=df_melted, x="Polarity Group", y="Value", hue="Metric", palette="Set2")
    plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    #plt.title(f"Error Metrics by Reference Score Polarity ({eval_label} vs {ref_label})", fontsize=12)
    plt.ylabel("Metric Score Value")
    plt.xlabel(f"Polarity group ({ref_label} scores)")
    plt.tight_layout()
    
    fig_name = f"polarity_errors_{ref_label}_vs_{eval_label}.pdf"
    plt.savefig(path.join(output_folder, fig_name), dpi=300)
    plt.close()


# =============================================================================
# 6. MAIN PIPELINE ORCHESTRATION
# =============================================================================

def print_metric_summary(metric_name: str, values: list):
    """Prints descriptive statistics for calculated metrics."""
    print(f"\n{metric_name} Summary:")
    if values:
        print(pd.Series(values).describe().round(2).to_frame().T)
    else:
        print("No valid scores computed.")


def main():
    output_folder = path.join("outputs")
    graph_folder = path.join(output_folder, "graphs")
    makedirs(graph_folder, exist_ok = True)

    # Load Domain Map
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
    print("1. ANNOTATION DENSITY, SPARSITY & RAW DISTRIBUTIONS")
    print("=" * 80)

    eval_cols = {
        "Teacher (OpenAI)": "openai_evaluation",
        "Student (Original)": "original_evaluation",
        "Human Annotators": "human_evaluation"
    }

    sparsity_df = analyze_nonzero_trait_counts(df, eval_cols)
    print(sparsity_df.to_string(index=False))
    sparsity_df.to_latex(path.join(output_folder, 'sparsity.tex'), index = False)

    plot_density_comparison(sparsity_df, graph_folder)
    plot_raw_score_distributions(df, eval_cols, graph_folder)

    coverage = compute_teacher_coverage_alignment(
        df, ref_col="openai_evaluation",
        target_cols={"Student": "original_evaluation", "Human": "human_evaluation"}
    )
    print("\n--- Teacher Trait Coverage / Active Recall ---")
    for k, v in coverage.items():
        print(f"  * {k} captures {v * 100:.1f}% of Teacher's active non-zero traits on average.")

    # 2. Row-Level Vector Metrics & Error Analysis
    print("\n" + "=" * 80)
    print("2. ROW-LEVEL VECTOR SIMILARITY & POLARITY ERROR ANALYSIS")
    print("=" * 80)

    comparisons = [
        ("openai_evaluation", "human_evaluation"),
        ("openai_evaluation", "original_evaluation"),
        ("original_evaluation", "human_evaluation"),
        ("original_evaluation", "openai_evaluation"),
    ]

    for id_comp, (ref_col, eval_col) in enumerate(comparisons):
        ref_label, eval_label = ref_col.replace('_evaluation', '').replace('original', 'OUR'), eval_col.replace('_evaluation', '').replace('original', 'OUR')

        print(f"\n{'='*50}\n({id_comp + 1}) {ref_label} vs {eval_label}\n{'='*50}")

        corr, cosine_sim, mse = compute_metrics_vs_teacher(
            df, col_ref=ref_col, col_eval=eval_col
        )
        print_metric_summary("Pearson Correlation", corr)
        print_metric_summary("Cosine Similarity", cosine_sim)
        print_metric_summary(f"MSE (vs {ref_label})", mse)

        # Polarity-based Error Breakdown
        polarity_err_df = analyze_errors_by_score_polarity(df, col_ref=ref_col, col_eval=eval_col)
        print(f"\n--- Error Analysis Grouped by Score Polarity ({eval_label} vs {ref_label}) ---")
        print(polarity_err_df.to_string(index=False))
        if 'human' in eval_label or 'human' in ref_label:
            polarity_err_df.to_latex(path.join(output_folder, f'polarity_{ref_label}_{eval_label}.tex'), index = False)

        # Attribute-wise Differences
        attr_diffs = compute_attribute_wise_differences(df, grouped_traits, col_ref=ref_col, col_eval=eval_col)

        # Generate Visualizations
        plot_attribute_differences(attr_diffs, ref_label, eval_label, graph_folder)
        plot_polarity_error_summary(polarity_err_df, ref_label, eval_label, graph_folder)

        # Export Excel and LaTeX Outputs
        excel_mode = 'w' if id_comp == 0 else "a"
        with pd.ExcelWriter(path.join(output_folder, 'attribute_diffs.xlsx'), mode=excel_mode) as xf:
            attr_diffs.to_excel(xf, sheet_name=f'{ref_label}_{eval_label}_attrs')
            polarity_err_df.to_excel(xf, sheet_name=f'{ref_label}_{eval_label}_polarity')

        if (ref_col == 'original_evaluation') and (eval_col == 'human_evaluation'):
            attr_diffs.round(2).drop(columns = 'Predominant Polarity').to_latex(path.join(output_folder, 'dff_original_human.tex'), index=False)

        print(f"\n--- Top 10 Attribute Differences ({eval_label} vs {ref_label}) ---")
        print(attr_diffs.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
