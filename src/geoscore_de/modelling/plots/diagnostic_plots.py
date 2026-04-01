import pandas as pd
import plotnine as gg


def build_predicted_vs_actual_plot(y_true, y_pred) -> gg.ggplot:
    """Create a predicted vs actual values plot.

    Args:
        y_true (array-like): The true target values.
        y_pred (array-like): The predicted target values from the model.

    Returns:
        gg.ggplot: A plotnine ggplot object showing predicted vs actual values.
    """
    try:
        df = pd.DataFrame({"Actual": y_true, "Predicted": y_pred})
        plot = (
            gg.ggplot(df, gg.aes(x="Actual", y="Predicted"))
            + gg.geom_point(alpha=0.6, size=1.8)
            # Add a reference line for perfect predictions
            + gg.geom_abline(intercept=0.0, slope=1.0, color="red", linetype="dashed")
            + gg.labs(x="Actual Values", y="Predicted Values", title="Predicted vs Actual Values")
            + gg.coord_equal()
            + gg.theme(aspect_ratio=1)
            + gg.theme_bw()
        )

        return plot
    except Exception as e:
        raise ValueError("Could not create predicted vs actual plot") from e


def build_residual_plot(y_true, y_pred) -> gg.ggplot:
    """Create a residuals plot.

    Args:
        y_true (array-like): The true target values.
        y_pred (array-like): The predicted target values from the model.

    Returns:
        gg.ggplot: A plotnine ggplot object showing residuals vs predicted values.
    """
    try:
        residuals = y_true - y_pred
        df = pd.DataFrame({"Predicted": y_pred, "Residuals": residuals})
        plot = (
            gg.ggplot(df, gg.aes(x="Predicted", y="Residuals"))
            + gg.geom_point(alpha=0.6, size=1.8)
            # Add a reference line at residual=0
            + gg.geom_hline(yintercept=0.0, color="red", linetype="dashed")
            + gg.labs(x="Predicted Values", y="Residuals", title="Residual Plot")
            + gg.theme_bw()
        )

        return plot
    except Exception as e:
        raise ValueError("Could not create residual plot") from e
