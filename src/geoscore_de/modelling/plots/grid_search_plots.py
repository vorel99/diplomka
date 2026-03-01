import pandas as pd
from matplotlib import pyplot as plt


def build_plot_grid_search_results(cv_results_df: pd.DataFrame, best_params: dict) -> None:
    """Create visualization of grid search results."""
    try:
        param_cols = [col for col in cv_results_df.columns if col.startswith("param_") and col != "params"]
        score_col = "mean_test_score" if "mean_test_score" in cv_results_df.columns else "mean_test_r2"

        if not param_cols:
            return

        n_params = len(param_cols)
        fig, axes = plt.subplots(1, min(n_params, 3), figsize=(6 * min(n_params, 3), 5))
        if n_params == 1:
            axes = [axes]

        for idx, param_col in enumerate(param_cols[:3]):
            ax = axes[idx]
            param_name = param_col.replace("param_", "")

            grouped = cv_results_df.groupby(param_col)[score_col].agg(["mean", "std"])

            x_values = grouped.index.astype(str)
            y_values = grouped["mean"]
            y_std = grouped["std"]

            ax.errorbar(x_values, y_values, yerr=y_std, marker="o", capsize=5, capthick=2, linewidth=2)

            best_value = str(best_params.get(param_name, ""))
            if best_value in x_values:
                best_idx = list(x_values).index(best_value)
                ax.plot(x_values[best_idx], y_values.iloc[best_idx], "r*", markersize=20, label="Best", zorder=5)

            ax.set_xlabel(param_name, fontsize=12, fontweight="bold")
            ax.set_ylabel("Mean CV Score (R²)", fontsize=12)
            ax.set_title(f"Impact of {param_name}", fontsize=13, fontweight="bold")
            ax.grid(True, alpha=0.3)
            ax.legend()

            if len(x_values) > 5:
                ax.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plot_path = "grid_search_param_importance.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

    except Exception as e:
        print(f"Warning: Could not create grid search visualization: {e}")
