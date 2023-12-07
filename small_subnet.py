import pandas as pd


def _has_complete_data(gauge_id):
    q_df = pd.read_csv(f"LamaH-CE/raw/D_gauges/2_timeseries//hourly/ID_{gauge_id}.csv",
                       sep=";", usecols=["YYYY", "qobs"])
    met_df = pd.read_csv(f"LamaH-CE/raw/B_basins_intermediate_all/2_timeseries/hourly/ID_{gauge_id}.csv",
                         sep=";", usecols=["YYYY"] + ["prec", "volsw_123",  "2m_temp", "surf_press"])
    if (q_df["qobs"] >= 0).all():
        q_df = q_df[(q_df["YYYY"] >= 2000) & (q_df["YYYY"] <= 2017)]
        met_df = met_df[(met_df["YYYY"] >= 2000) & (met_df["YYYY"] <= 2017)]
        if len(q_df) == (18 * 365 + 5) * 24 and len(met_df) == (18 * 365 + 5) * 24:  # number of hours in 2000-2017
            return True
    return False


def _collect_upstream(gauge_id, adj_df):
    collected_ids = set()
    is_complete = _has_complete_data(gauge_id)
    if is_complete:
        collected_ids.add(gauge_id)
        predecessor_ids = set(adj_df[adj_df["NEXTDOWNID"] == gauge_id]["ID"])
        collected_ids.update(*[_collect_upstream(pred_id, adj_df) for pred_id in predecessor_ids])
    return collected_ids


adj_df = pd.read_csv("B_basins_intermediate_all/1_attributes/Stream_dist.csv", sep=";")
connected_gauges = set(adj_df["ID"]).union(adj_df["NEXTDOWNID"])
for base_gauge_id in connected_gauges:
    subnet = list(_collect_upstream(base_gauge_id, adj_df))
    if len(subnet) <= 10:
        print(f"base {base_gauge_id} -> {subnet}")
    if base_gauge_id == 399:
        print(f"should be 375: {len(subnet)}")
