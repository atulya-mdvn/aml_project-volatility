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
    "p3_text_pc1",
    "p3_text_pc2",
    "p3_text_pc3",
    "p3_text_pc4",
    "p3_text_pc5",
    "p3_text_pc6",
    "p3_text_pc7",
    "p3_text_pc8",
    "p3_text_pc9",
    "p3_text_pc10",
    "p3_text_pc11",
    "p3_text_pc12",
    "p3_text_pc13",
    "p3_text_pc14",
    "p3_text_pc15",
    "p3_text_pc16",
    "p3_text_pc17",
    "p3_text_pc18",
    "p3_text_pc19",
    "p3_text_pc20",
]

P4_FEATURE_COLUMNS = [
    "date",
    "p4_graph_num_nodes",
    "p4_graph_num_edges",
    "p4_graph_density",
    "p4_graph_avg_abs_corr",
    "p4_graph_max_abs_corr",
    "p4_graph_min_corr",
    "p4_graph_corr_dispersion",
    "p4_graph_pc1",
    "p4_graph_pc2",
    "p4_graph_pc3",
    "p4_graph_pc4",
    "p4_graph_pc5",
    "p4_graph_pc6",
    "p4_graph_pc7",
    "p4_graph_pc8",
    "p4_graph_pc9",
    "p4_graph_pc10",
    "p4_graph_pc11",
    "p4_graph_pc12",
    "p4_graph_pc13",
    "p4_graph_pc14",
    "p4_graph_pc15",
    "p4_graph_pc16",
    "p4_graph_pc17",
    "p4_graph_pc18",
    "p4_graph_pc19",
    "p4_graph_pc20",
]