import marimo

__generated_with = "0.22.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    from glob import glob

    return glob, pd


@app.cell
def _(glob, pd):
    files = glob("unproc/atp_matches_[12]*.csv")
    dfs = [pd.read_csv(file) for file in files]
    df = pd.concat(dfs)
    return (df,)


@app.cell
def _(df):
    df.columns
    return


@app.cell
def _(df):
    df_1 = df[
        (df["tourney_level"] != "O")
        & (df["score"] != "UNK")
        & (df["loser_entry"] != "S")
    ].dropna(subset=["score", "winner_hand", "loser_hand", "w_svpt", "l_svpt"])
    return (df_1,)


@app.cell
def _(df_1):
    rename = {
        "tourney_id": "tournament_id",
        "tourney_name": "tournament_name",
        "surface": "surface",
        "draw_size": "draw_size",
        "tourney_level": "tournament_tier",
        "tourney_date": "tournament_start_date",
        "minutes": "duration",
        "round": "round",
        "best_of": "best_of",
        "score": "score",
        "winner_id": "W_id",
        "winner_entry": "W_entry",
        "winner_name": "W_name",
        "winner_hand": "W_hand",
        "winner_ht": "W_height",
        "winner_ioc": "W_country",
        "winner_age": "W_age",
        "winner_rank": "W_rank",
        "winner_rank_points": "W_rank_points",
        "w_ace": "W_aces",
        "w_df": "W_double_faults",
        "w_svpt": "W_total_serve_pts",
        "w_1stIn": "W_first_serve_in",
        "w_1stWon": "W_first_serve_won",
        "w_2ndWon": "W_second_serve_won",
        "w_SvGms": "W_service_games",
        "w_bpSaved": "W_break_pts_saved",
        "w_bpFaced": "W_break_pts_faced",
        "loser_id": "L_id",
        "loser_entry": "L_entry",
        "loser_name": "L_name",
        "loser_hand": "L_hand",
        "loser_ht": "L_height",
        "loser_ioc": "L_country",
        "loser_age": "L_age",
        "loser_rank": "L_rank",
        "loser_rank_points": "L_rank_points",
        "l_ace": "L_aces",
        "l_df": "L_double_faults",
        "l_svpt": "L_total_serve_pts",
        "l_1stIn": "L_first_serve_in",
        "l_1stWon": "L_first_serve_won",
        "l_2ndWon": "L_second_serve_won",
        "l_SvGms": "L_service_games",
        "l_bpSaved": "L_break_pts_saved",
        "l_bpFaced": "L_break_pts_faced",
    }

    df_2 = (
        df_1[rename.keys()]
        .rename(columns=rename)
        .sort_index(axis=1, ascending=False)
    ).reset_index(drop=True)
    return (df_2,)


@app.cell
def _(df_2):
    df_2.sample(10)
    return


@app.cell
def _():
    import re


    def preproc_score(value):
        if "?" in value:
            return None
        if "unfinished" in value or "abandoned" in value:
            return [["Abandoned"]]
        if "Walkover" in value:
            return [["Walkover"]]
        parts = value.strip().split()
        struct = []
        for part in parts:
            item = []
            if part == "RET":
                item.append("Retired")
            elif part == "W/O":
                item.append("Walkover")
            elif part == "DEF" or part == "Def." or part == "Default":
                item.append("Default")
            elif part == "ABD":
                item.append("Abandoned")
            elif "[" in part:
                m = re.match(r"\[(\d+)-(\d+)\]", part)
                W, L = m.groups()
                item.append(int(W))
                item.append(int(L))
                item.append("SuperTiebreak")
            else:
                m = re.match(r"(\d+)-(\d+)(\(\d+\))?", part)
                W, L, tiebreak = m.groups()
                item.append(int(W))
                item.append(int(L))
                if tiebreak:
                    item.append("Tiebreak")
                    item.append(int(tiebreak[1:-1]))
            struct.append(item)

        return struct

    return (preproc_score,)


@app.function
def preproc_tier(value):
    return {
        "A": "OtherAtp",
        "G": "GrandSlam",
        "M": "Masters",
        "F": "Finals",
        "D": "DavisCup",
    }[value]


@app.cell
def _():
    from datetime import datetime


    def preproc_date(value):
        value = str(value)
        year = int(value[:4])
        month = int(value[4:6])
        day = int(value[6:])
        return datetime(year, month, day)

    return (preproc_date,)


@app.function
def preproc_round(value):
    return {
        "R128": "R128",
        "R64": "R64",
        "R32": "R32",
        "R16": "R16",
        "QF": "Quarterfinal",
        "SF": "Semifinal",
        "F": "Final",
        "RR": "Robin",
        "ER": "Early",
        "BR": "Bronze",
    }[value]


@app.cell
def _():
    import numpy as np


    def preproc_hand(value):
        return {"R": "Right", "L": "Left", "A": "Both", "U": np.nan}[value]

    return np, preproc_hand


@app.cell
def _(np):
    def preproc_entry(value):
        return {
            np.nan: "DirectAccept",
            "Q": "Qualifier",
            "WC": "WildCard",
            "Q": "Qualifier",
            "LL": "LuckyLoser",
            "PR": "ProtectRank",
            "SE": "SpecialExempt",
            "ALT": "Alternate",
            "Alt": "Alternate",
            "W": "Walkover",
        }[value]

    return (preproc_entry,)


@app.function
def preproc_tournament_name(value):
    value = value.replace("-", " ")
    value = value.replace("St.", "St")
    value = value.replace(" De ", " de ")
    value = value.replace(" Of ", " of ")
    for tire in [
        " Olympics",
        " Indoor",
        " Outdoor",
        " WCT",
        " Masters",
        " Indoor WCT",
        " 1",
        " 2",
        " 3",
        " NTL",
        " Final",
        " SF",
    ]:
        value = value.replace(tire, "")
    if "Davis" in value:
        value = value.split(":")[0]
        value = " ".join(value.split()[:-2])
    if value == "Montreal / Toronto":
        return "Montreal"
    if value == "Australian Chps.":
        return "Australian Championships"
    if value == "s Hertogenbosch":
        return "Hertogenbosch"
    if value == "Pittsburghs":
        return "Pittsburgh"
    value = value.strip()
    value = value.replace("Atp", "ATP")
    return value


@app.cell
def _(df_2, preproc_date, preproc_entry, preproc_hand, preproc_score):
    df_2["tournament_tier"] = df_2["tournament_tier"].map(preproc_tier)
    df_2["tournament_name"] = df_2["tournament_name"].map(preproc_tournament_name)
    df_2["tournament_start_date"] = df_2["tournament_start_date"].map(preproc_date)
    df_2["score"] = df_2["score"].map(preproc_score)
    df_2["round"] = df_2["round"].map(preproc_round)
    df_2["W_hand"] = df_2["W_hand"].map(preproc_hand)
    df_2["L_hand"] = df_2["L_hand"].map(preproc_hand)
    df_2["W_entry"] = df_2["W_entry"].map(preproc_entry)
    df_2["L_entry"] = df_2["L_entry"].map(preproc_entry)
    return


@app.cell
def _(df_2):
    df_2.sample(10)
    return


@app.cell
def _(df_2):
    len(df_2)
    return


@app.cell
def _(df_2):
    df_2
    return


@app.cell
def _(df_2):
    df_2.sort_values(by="tournament_start_date", inplace=True)
    return


@app.cell
def _(df_2):
    df_3 = df_2.convert_dtypes()
    return (df_3,)


@app.cell
def _(df_3):
    df_3.to_csv("input.csv", index=False)
    return


@app.cell
def _(df_3):
    df_3
    return


if __name__ == "__main__":
    app.run()
