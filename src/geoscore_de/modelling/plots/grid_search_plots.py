import pandas as pd
import plotnine as gg


def build_plot_grid_search_results(cv_results_df: pd.DataFrame, best_params: dict) -> gg.ggplot | None:
    """Create visualization of grid search results."""
    try:
        param_cols = [col for col in cv_results_df.columns if col.startswith("param_") and col != "params"]
        score_col = "mean_test_score" if "mean_test_score" in cv_results_df.columns else "mean_test_r2"

        if not param_cols:
            return

        plots: list[gg.ggplot] = []

        for param_col in param_cols[:3]:
            param_name = param_col.replace("param_", "")

            grouped = cv_results_df.groupby(param_col)[score_col].agg(["mean", "std"]).reset_index()
            numeric_param_values = pd.to_numeric(grouped[param_col], errors="coerce")
            if numeric_param_values.notna().all():
                grouped = grouped.assign(_param_numeric=numeric_param_values).sort_values("_param_numeric")
            else:
                grouped = grouped.sort_values(by=param_col, key=lambda series: series.astype(str))

            grouped["param_value"] = grouped[param_col].astype(str)
            grouped["param_value"] = pd.Categorical(
                grouped["param_value"],
                categories=grouped["param_value"].tolist(),
                ordered=True,
            )
            grouped["y_min"] = grouped["mean"] - grouped["std"]
            grouped["y_max"] = grouped["mean"] + grouped["std"]

            best_value = str(best_params.get(param_name, ""))
            best_points = grouped[grouped["param_value"] == best_value]

            plot = (
                gg.ggplot(grouped, gg.aes(x="param_value", y="mean"))
                + gg.geom_line(group=1)
                + gg.geom_point(size=2)
                + gg.geom_errorbar(gg.aes(ymin="y_min", ymax="y_max"), width=0.2)
                + gg.labs(
                    x=param_name,
                    y="Mean CV Score (R²)",
                    title=f"Impact of {param_name}",
                )
                + gg.theme_bw()
                + gg.theme(figure_size=(6, 5))
            )

            if len(grouped) > 5:
                plot += gg.theme(axis_text_x=gg.element_text(rotation=45, ha="right"))

            if not best_points.empty:
                plot += gg.geom_point(
                    data=best_points,
                    mapping=gg.aes(x="param_value", y="mean"),
                    color="red",
                    shape="*",
                    size=5,
                )

            plots.append(plot)

        if not plots:
            return

        combined_plot = plots[0]
        for plot in plots[1:]:
            combined_plot = combined_plot | plot

        return combined_plot

    except Exception as e:
        print(f"Warning: Could not create grid search visualization: {e}")
