import marimo

__generated_with = "0.22.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd

    return mo, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's try to predict age of winners and losers.

    Why not? I don't have any ideas yet what else we could predict.

    I won't even do any feature engineering...
    """)
    return


@app.cell
def _(pd):
    df = pd.read_csv("input.csv")
    df.dropna(subset=["W_age", "L_age"], inplace=True)
    return (df,)


@app.cell
def _(df, pd):
    df["tournament_start_date"] = pd.to_datetime(df["tournament_start_date"])
    return


@app.cell
def _():
    cat_features = [
        "tournament_tier",
        "tournament_name",
        "surface",
        "round",
        "draw_size",
        "best_of",
        "W_country",
        "W_hand",
        "W_entry",
        "L_country",
        "L_hand",
        "L_entry",
    ]

    num_features = [
        "duration",
        "W_total_serve_pts",
        "W_service_games",
        "W_second_serve_won",
        "W_rank_points",
        "W_rank",
        "W_height",
        "W_first_serve_won",
        "W_first_serve_in",
        "W_double_faults",
        "W_break_pts_saved",
        "W_break_pts_faced",
        "W_aces",
        "L_total_serve_pts",
        "L_service_games",
        "L_second_serve_won",
        "L_rank_points",
        "L_rank",
        "L_height",
        "L_first_serve_won",
        "L_first_serve_in",
        "L_double_faults",
        "L_break_pts_saved",
        "L_break_pts_faced",
        "L_aces",
    ]

    features = cat_features + num_features
    return cat_features, features


@app.cell
def _(df, features):
    import numpy as np

    X = df[features].fillna(0)
    y = df[["W_age"]].to_numpy()
    return X, y


@app.cell
def _(y):
    y
    return


@app.cell
def _(X, y):
    from sklearn.model_selection import train_test_split

    X_train, X_eval, y_train, y_eval = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=42
    )
    return X_eval, X_train, y_eval, y_train


@app.cell
def _(cat_features):
    from catboost import CatBoostRegressor

    model = CatBoostRegressor(cat_features=cat_features)
    return (model,)


@app.cell
def _(X_eval, X_train, model, y_eval, y_train):
    model.fit(X_train, y_train, eval_set=(X_eval, y_eval))
    return


@app.cell
def _(X, model):
    dict(
        sorted(
            zip(X.columns, model.feature_importances_),
            key=lambda p: p[1],
            reverse=True,
        )
    )
    return


if __name__ == "__main__":
    app.run()
