"""
FedLineage — model lineage graph and simple root-cause helpers.

Builds Client -> Update -> Round -> Global Model relationships
programmatically from the simulated FL metadata (no hand-built nodes).
"""

import json
import os

import networkx as nx
import pandas as pd
import plotly.graph_objects as go

DATA_DIR = "data"

CLIENT_COLORS = {
    "C1": "#6378FF",
    "C2": "#00D9B5",
    "C3": "#FF5C6C",
    "C4": "#C46BFF",
    "C5": "#FFB347",
}

ACCENT_BLUE = "#6378FF"
ACCENT_TEAL = "#00D9B5"
ACCENT_RED = "#FF5C6C"


def client_node(cid):
    return f"Client_{cid}"


def update_node(cid, round_num):
    return f"Update_{cid}_R{round_num}"


def round_node(round_num):
    return f"Round_{round_num}"


def model_node(round_num):
    return f"Global_Model_V{round_num}"


def load_tables(data_dir=DATA_DIR):
    clients = pd.read_csv(os.path.join(data_dir, "clients.csv"))
    rounds = pd.read_csv(os.path.join(data_dir, "rounds.csv"))
    return clients, rounds


def build_lineage_graph(clients, rounds):
    """Create a directed lineage graph for every client x round."""
    G = nx.DiGraph()
    client_ids = sorted(clients["client_id"].unique().tolist())
    round_nums = sorted(rounds["round"].astype(int).unique().tolist())

    for cid in client_ids:
        G.add_node(client_node(cid), kind="client", client_id=cid, label=cid)

    for r in round_nums:
        rrow = rounds.loc[rounds["round"] == r].iloc[0]
        G.add_node(
            round_node(r),
            kind="round",
            round=int(r),
            label=f"Round {r}",
        )
        G.add_node(
            model_node(r),
            kind="global_model",
            round=int(r),
            label=f"Global Model V{r}",
            global_accuracy=float(rrow["global_accuracy"]),
            global_loss=float(rrow["global_loss"]),
        )
        G.add_edge(round_node(r), model_node(r), relation="produces")
        if r > min(round_nums):
            G.add_edge(model_node(r - 1), round_node(r), relation="continues")

        for cid in client_ids:
            crow = clients[(clients["client_id"] == cid) & (clients["round"] == r)]
            if crow.empty:
                continue
            row = crow.iloc[0]
            uid = update_node(cid, r)
            G.add_node(
                uid,
                kind="update",
                client_id=cid,
                round=int(r),
                label=f"{cid} Update",
                status=str(row["status"]),
                cpu_usage=float(row["cpu_usage"]),
                memory_usage=float(row["memory_usage"]),
                network_bandwidth=float(row["network_bandwidth"]),
                training_time=float(row["training_time"]),
                local_loss=float(row["local_loss"]),
                local_accuracy=float(row["local_accuracy"]),
            )
            G.add_edge(client_node(cid), uid, relation="trains")
            G.add_edge(uid, round_node(r), relation="contributes")

    return G


def lineage_payload(G, clients, rounds):
    client_ids = sorted(clients["client_id"].unique().tolist())
    models = []
    for r in sorted(rounds["round"].astype(int).unique().tolist()):
        rrow = rounds.loc[rounds["round"] == r].iloc[0]
        models.append({
            "model": f"Global_V{r}",
            "round": int(r),
            "clients": client_ids,
            "global_accuracy": float(rrow["global_accuracy"]),
            "global_loss": float(rrow["global_loss"]),
        })

    nodes = []
    for nid, data in G.nodes(data=True):
        item = {"id": nid, "kind": data.get("kind"), "label": data.get("label", nid)}
        for key in ("client_id", "round", "status", "global_accuracy"):
            if key in data:
                item[key] = data[key]
        nodes.append(item)

    edges = [{"source": u, "target": v, "relation": d.get("relation", "")}
             for u, v, d in G.edges(data=True)]

    anomaly = find_anomaly(clients, rounds)
    return {"models": models, "nodes": nodes, "edges": edges, "anomaly": anomaly}


def find_anomaly(clients, rounds):
    flagged = clients[clients["status"] == "straggler"]
    if flagged.empty:
        return None
    row = flagged.sort_values("round").iloc[-1]
    r = int(row["round"])
    rrow = rounds.loc[rounds["round"] == r].iloc[0]
    prev = rounds.loc[rounds["round"] == r - 1]
    prev_acc = float(prev.iloc[0]["global_accuracy"]) if not prev.empty else None
    return {
        "client": row["client_id"],
        "round": r,
        "model": f"Global_V{r}",
        "cpu_usage": float(row["cpu_usage"]),
        "memory_usage": float(row["memory_usage"]),
        "network_bandwidth": float(row["network_bandwidth"]),
        "training_time": float(row["training_time"]),
        "local_loss": float(row["local_loss"]),
        "local_accuracy": float(row["local_accuracy"]),
        "status": "Potential contributing straggler",
        "global_accuracy": float(rrow["global_accuracy"]),
        "previous_global_accuracy": prev_acc,
    }


def write_lineage_json(payload, data_dir=DATA_DIR):
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "lineage.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def subgraph_nodes(G, round_num):
    """Nodes needed to show Client -> Update -> Round -> Global Model for one version."""
    keep = {round_node(round_num), model_node(round_num)}
    for nid, data in G.nodes(data=True):
        if data.get("kind") == "update" and int(data.get("round", -1)) == int(round_num):
            keep.add(nid)
            keep.add(client_node(data["client_id"]))
    return G.subgraph(keep).copy()


def lineage_figure(G, round_num):
    """Plotly figure for one model version. Anomalous client updates are highlighted."""
    sub = subgraph_nodes(G, round_num)
    updates = sorted(
        [n for n, d in sub.nodes(data=True) if d.get("kind") == "update"],
        key=lambda n: sub.nodes[n]["client_id"],
    )
    n_u = max(len(updates), 1)
    xs = [(i - (n_u - 1) / 2) for i in range(n_u)]

    pos = {
        model_node(round_num): (0.0, 3.0),
        round_node(round_num): (0.0, 2.0),
    }
    for i, uid in enumerate(updates):
        cid = sub.nodes[uid]["client_id"]
        pos[uid] = (xs[i], 1.0)
        pos[client_node(cid)] = (xs[i], 0.0)

    edge_x, edge_y = [], []
    for u, v in sub.edges():
        if u not in pos or v not in pos:
            continue
        edge_x += [pos[u][0], pos[v][0], None]
        edge_y += [pos[u][1], pos[v][1], None]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(color="rgba(139,157,195,0.45)", width=1.6),
        hoverinfo="none", showlegend=False,
    ))

    for kind, size in (("global_model", 28), ("round", 22), ("update", 18), ("client", 16)):
        ids = [n for n, d in sub.nodes(data=True) if d.get("kind") == kind and n in pos]
        if not ids:
            continue
        colors, texts, hovers, outlines = [], [], [], []
        for nid in ids:
            d = sub.nodes[nid]
            is_bad = d.get("status") == "straggler"
            if kind == "update" and is_bad:
                colors.append(ACCENT_RED)
                texts.append(f"{d['client_id']} Update ⚠")
            elif kind == "client":
                colors.append(CLIENT_COLORS.get(d.get("client_id"), ACCENT_BLUE))
                texts.append(d.get("label", nid))
            elif kind == "global_model":
                colors.append(ACCENT_TEAL)
                texts.append(d.get("label", nid))
            else:
                colors.append(ACCENT_BLUE)
                texts.append(d.get("label", nid))

            outlines.append(ACCENT_RED if is_bad else "#0D1117")
            if kind == "update":
                hovers.append(
                    f"<b>{d.get('label')}</b><br>"
                    f"Status: {d.get('status')}<br>"
                    f"CPU: {d.get('cpu_usage')}%<br>"
                    f"Memory: {d.get('memory_usage')}%<br>"
                    f"Bandwidth: {d.get('network_bandwidth')} MB/s<br>"
                    f"Train time: {d.get('training_time')}s"
                )
            elif kind == "global_model":
                hovers.append(
                    f"<b>{d.get('label')}</b><br>"
                    f"Accuracy: {d.get('global_accuracy', '—')}%"
                )
            else:
                hovers.append(f"<b>{d.get('label', nid)}</b>")

        fig.add_trace(go.Scatter(
            x=[pos[n][0] for n in ids],
            y=[pos[n][1] for n in ids],
            mode="markers+text",
            marker=dict(
                size=size,
                color=colors,
                line=dict(width=2, color=outlines),
                opacity=0.95,
            ),
            text=texts,
            textposition="top center",
            textfont=dict(size=11, color="#E2E8F0"),
            hovertemplate="%{customdata}<extra></extra>",
            customdata=hovers,
            showlegend=False,
        ))

    fig.update_layout(
        paper_bgcolor="rgba(16,20,38,0.0)",
        plot_bgcolor="rgba(16,20,38,0.0)",
        font=dict(family="Inter,sans-serif", color="#CBD5E1", size=12),
        margin=dict(l=24, r=24, t=24, b=24),
        height=460,
        xaxis=dict(visible=False, range=[min(xs) - 1.1, max(xs) + 1.1] if xs else [-2, 2]),
        yaxis=dict(visible=False, range=[-0.45, 3.55]),
        hoverlabel=dict(bgcolor="#1A2035", bordercolor=ACCENT_BLUE, font_color="#E2E8F0"),
    )
    return fig


def main():
    clients, rounds = load_tables()
    G = build_lineage_graph(clients, rounds)
    payload = lineage_payload(G, clients, rounds)
    path = write_lineage_json(payload)
    anomaly = payload["anomaly"]
    print(f"Nodes      : {G.number_of_nodes()}")
    print(f"Edges      : {G.number_of_edges()}")
    print(f"Models     : {len(payload['models'])}")
    print(f"Wrote      : {path}")
    if anomaly:
        print(f"Anomaly    : {anomaly['client']} in round {anomaly['round']} ({anomaly['model']})")


if __name__ == "__main__":
    main()
