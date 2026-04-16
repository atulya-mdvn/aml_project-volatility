"""
Schemas for pillar outputs.

Each pillar must export a daily feature table with:
- column 'date'
- one row per date
- no duplicate dates
- no target variables

These files are later merged in the fusion stage.
"""


P3_FEATURE_COLUMNS = [
    "date",
    "p3_doc_count",
    "p3_fed_count",
    "p3_news_count",
    "p3_earnings_count",
    "p3_sent_pos_mean",
    "p3_sent_neg_mean",
    "p3_sent_neu_mean",
    "p3_sent_pos_std",
    "p3_sent_neg_std",
    "p3_sent_neu_std",
    "p3_uncertainty_mean",
    "p3_uncertainty_std",
] + [f"p3_text_pc{i}" for i in range(1, 21)]

P3_DOCUMENT_COLUMNS = [
    "doc_id",
    "date",
    "timestamp",
    "source",
    "source_type",
    "title",
    "text",
    "ticker",
    "url",
]

P3_DOC_FEATURE_COLUMNS = [
    "doc_id",
    "date",
    "source_type",
    "p3_sent_pos",
    "p3_sent_neg",
    "p3_sent_neu",
    "p3_uncertainty_score",
] + [f"p3_emb_{i}" for i in range(1, 769)]

P4_FEATURE_COLUMNS = [
    "date",
    "p4_graph_num_nodes",
    "p4_graph_num_edges",
    "p4_graph_density",
    "p4_graph_avg_abs_corr",
    "p4_graph_max_abs_corr",
    "p4_graph_min_corr",
    "p4_graph_corr_dispersion",
] + [f"p4_graph_pc{i}" for i in range(1, 21)]

P4_NODE_FEATURE_COLUMNS = [
    "date",
    "asset",
    "asset_class",
    "close",
    "volume",
    "p4_ret_1d",
    "p4_ret_5d",
    "p4_vol_20d",
    "p4_momentum_20d",
    "p4_volume_z",
]

P4_EDGE_FEATURE_COLUMNS = [
    "date",
    "asset_i",
    "asset_j",
    "p4_corr_60d",
    "p4_abs_corr_60d",
]