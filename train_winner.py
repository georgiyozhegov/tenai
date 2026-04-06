import marimo

__generated_with = "0.22.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np

    return np, pd


@app.cell
def _(pd):
    df = pd.read_csv("input.csv")
    return (df,)


@app.cell
def _(df):
    df.sample(10)
    return


@app.cell
def _(df):
    for side in ["W", "L"]:
        df[f"{side}_first_serve_won_pct"] = (
            df[f"{side}_first_serve_won"] / df[f"{side}_first_serve_in"]
        )
        df.drop(
            columns=[f"{side}_first_serve_won", f"{side}_first_serve_in"],
            inplace=True,
        )

        df[f"{side}_break_pts_saved_pct"] = (
            df[f"{side}_break_pts_saved"] / df[f"{side}_break_pts_faced"]
        )
        df.drop(
            columns=[f"{side}_break_pts_saved", f"{side}_break_pts_faced"],
            inplace=True,
        )
    return


@app.cell
def _(df):
    df
    return


@app.cell
def _():
    avg_features = [
        "total_serve_pts",
        "service_games",
        "second_serve_won",
        "first_serve_won_pct",
        "double_faults",
        "break_pts_saved_pct",
        "aces",
    ]
    return (avg_features,)


@app.cell
def _(df, pd):
    W_df = df[[c for c in df.columns if c.startswith("W")]].rename(
        columns=lambda c: c[2:]
    )
    L_df = df[[c for c in df.columns if c.startswith("L")]].rename(
        columns=lambda c: c[2:]
    )
    player_df = pd.concat([W_df, L_df], axis=0)
    return (player_df,)


@app.cell
def _(avg_features, player_df):
    avg_df = (
        player_df.groupby("id")[avg_features]
        .mean()
        .rename(columns=lambda c: f"avg_{c}")
    )
    avg_df.head()
    return (avg_df,)


@app.cell
def _(avg_df, df):
    df_with_avg = df.merge(
        avg_df.add_prefix("W_"), left_on="W_id", right_on="id"
    ).merge(avg_df.add_prefix("L_"), left_on="L_id", right_on="id")
    return (df_with_avg,)


@app.cell
def _(avg_features, df_with_avg):
    W_columns = [f"W_{c}" for c in avg_features]
    W_avg_columns = df_with_avg.columns[df_with_avg.columns.str.contains("W_avg_")]

    for orig_c, avg_c in zip(W_columns, W_avg_columns):
        df_with_avg[f"{orig_c}_avg_diff"] = (
            df_with_avg[orig_c] - df_with_avg[avg_c]
        )

    df_with_avg.drop(columns=W_avg_columns, inplace=True)

    L_columns = [f"L_{c}" for c in avg_features]
    L_avg_columns = df_with_avg.columns[df_with_avg.columns.str.contains("L_avg_")]

    for orig_c, avg_c in zip(L_columns, L_avg_columns):
        df_with_avg[f"{orig_c}_avg_diff"] = (
            df_with_avg[orig_c] - df_with_avg[avg_c]
        )

    df_with_avg.drop(columns=L_avg_columns, inplace=True)
    return


@app.cell
def _(df_with_avg):
    df_with_avg
    return


@app.cell
def _(df_with_avg, np):
    player_columns = [c[2:] for c in df_with_avg.columns if c.startswith("W_")]

    W_features = [f"W_{c}" for c in player_columns]
    L_features = [f"L_{c}" for c in player_columns]

    mask = np.random.random(len(df_with_avg)) <= 0.5

    df_with_avg.loc[mask, [f"A_{c}" for c in player_columns]] = df_with_avg.loc[
        mask, L_features
    ].to_numpy()
    df_with_avg.loc[mask, [f"B_{c}" for c in player_columns]] = df_with_avg.loc[
        mask, W_features
    ].to_numpy()

    df_with_avg.loc[~mask, [f"A_{c}" for c in player_columns]] = df_with_avg.loc[
        ~mask, W_features
    ].to_numpy()
    df_with_avg.loc[~mask, [f"B_{c}" for c in player_columns]] = df_with_avg.loc[
        ~mask, L_features
    ].to_numpy()

    df_with_avg["target"] = mask.astype(int)

    df_with_avg.drop(columns=W_features + L_features, inplace=True)
    return (player_columns,)


@app.cell
def _(player_columns):
    CAT = [
        "tournament_tier",
        "tournament_name",
        "surface",
        "round",
        "best_of",
        "A_entry",
        "A_country",
        "A_hand",
        "B_entry",
        "B_country",
        "B_hand",
    ]

    NUM = (
        [
            "duration",
            "draw_size",
        ]
        + [
            f"A_{c}"
            for c in player_columns
            if (f"A_{c}" not in CAT) and (c not in {"id", "name"})
        ]
        + [
            f"B_{c}"
            for c in player_columns
            if (f"B_{c}" not in CAT) and (c not in {"id", "name"})
        ]
    )
    return CAT, NUM


@app.cell
def _(CAT, NUM, df_with_avg):
    X = df_with_avg[CAT + NUM].fillna(0)
    y = df_with_avg[["target"]]
    return X, y


@app.cell
def _(X, y):
    from sklearn.model_selection import train_test_split

    X_train, X_eval, y_train, y_eval = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=55,
        shuffle=True,
        stratify=y,
    )
    return X_eval, X_train, y_eval, y_train


@app.cell
def _(CAT, X_eval, X_train, y_eval, y_train):
    from catboost import CatBoostClassifier

    model = CatBoostClassifier(
        cat_features=CAT, l2_leaf_reg=0.01, eval_metric="AUC"
    )
    model.fit(X_train, y_train, eval_set=(X_eval, y_eval))
    return (model,)


@app.cell
def _(X_eval, model):
    y_pred = model.predict_proba(X_eval)[:, 1]
    return (y_pred,)


@app.cell
def _(CAT, NUM, df_with_avg, model):
    def predict_sample():
        X_sample = df_with_avg.sample()
        print(f"{X_sample['A_name'].values[0]} vs {X_sample['B_name'].values[0]}")
        prob = model.predict_proba(X_sample[CAT + NUM])[0][1]
        if prob <= 0.5:
            winner = X_sample["A_name"].values[0]
        else:
            winner = X_sample["B_name"].values[0]
        print(f"The winner is {winner} (P = {prob:.2f})")

    return (predict_sample,)


@app.cell
def _(predict_sample):
    predict_sample()
    return


@app.cell
def _(y_pred):
    import matplotlib.pyplot as plt

    plt.plot(sorted(y_pred))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
