"""Account-device graph features, built with networkx on time snapshots.

Accounts and device fingerprints are nodes; an edge means the account was seen
on that device. Fraud rings show up as components where many accounts and
cards share a few devices. A snapshot taken at day D only uses edges from
before day D, and is applied to transactions from day D onwards.
"""
from __future__ import annotations

import networkx as nx
import pandas as pd


def snapshot(df: pd.DataFrame, before_day: int) -> dict:
    past = df.loc[(df["day"] < before_day) & df["device_fp"].notna(), ["uid", "device_fp", "card1"]]
    edges = past.drop_duplicates(["uid", "device_fp"])
    g = nx.Graph()
    g.add_edges_from((("u", u), ("d", d)) for u, d in zip(edges["uid"], edges["device_fp"]))
    card_of = dict(zip(past["uid"], past["card1"]))
    uid_feats, dev_feats = {}, {}
    for comp in nx.connected_components(g):
        uids = [n[1] for n in comp if n[0] == "u"]
        n_cards = len({card_of[u] for u in uids})
        for u in uids:
            uid_feats[u] = (len(uids), n_cards, g.degree(("u", u)))
        for n in comp:
            if n[0] == "d":
                dev_feats[n[1]] = g.degree(n)
    return {"uid": uid_feats, "dev": dev_feats, "nodes": g.number_of_nodes(),
            "edges": g.number_of_edges()}


def monthly_snapshots(df: pd.DataFrame, starts: list[int]) -> dict[int, dict]:
    return {d: snapshot(df, d) for d in starts}
