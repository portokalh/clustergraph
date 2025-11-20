#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 11:55:25 2025

@author: alex
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate node_region_labels.csv
Case: EXACTLY 1 node per cortical region (68 total)

Node ordering = row ordering of first QSM CSV file
Region list provided by Alexandra (68 labels)
"""

import os
import glob
import pandas as pd

# --------------------------------------------------------
# Paths
# --------------------------------------------------------
ROOT = "/mnt/newStor/paros/paros_WORK/alex/alex4gaudi/GAUDI-implementation"
QSM_NODES_DIR = os.path.join(ROOT, "processed_graph_data_110325", "qsm_nodes")

OUTFILE = os.path.join(
    ROOT,
    "columns4gaudi111825",
    "utilities",
    "node_region_labels.csv"
)

# --------------------------------------------------------
# 68-region list (exact Alex order)
# --------------------------------------------------------
regions = [
'lh_bankssts','lh_caudalanteriorcingulate','lh_caudalmiddlefrontal','lh_cuneus',
'lh_entorhinal','lh_fusiform','lh_inferiorparietal','lh_inferiortemporal',
'lh_isthmuscingulate','lh_lateraloccipital','lh_lateralorbitofrontal','lh_lingual',
'lh_medialorbitofrontal','lh_middletemporal','lh_parahippocampal','lh_paracentral',
'lh_parsopercularis','lh_parsorbitalis','lh_parstriangularis','lh_pericalcarine',
'lh_postcentral','lh_posteriorcingulate','lh_precentral','lh_precuneus',
'lh_rostralanteriorcingulate','lh_rostralmiddlefrontal','lh_superiorfrontal',
'lh_superiorparietal','lh_superiortemporal','lh_supramarginal','lh_frontalpole',
'lh_temporalpole','lh_transversetemporal','lh_insula',
'rh_bankssts','rh_caudalanteriorcingulate','rh_caudalmiddlefrontal','rh_cuneus',
'rh_entorhinal','rh_fusiform','rh_inferiorparietal','rh_inferiortemporal',
'rh_isthmuscingulate','rh_lateraloccipital','rh_lateralorbitofrontal','rh_lingual',
'rh_medialorbitofrontal','rh_middletemporal','rh_parahippocampal','rh_paracentral',
'rh_parsopercularis','rh_parsorbitalis','rh_parstriangularis','rh_pericalcarine',
'rh_postcentral','rh_posteriorcingulate','rh_precentral','rh_precuneus',
'rh_rostralanteriorcingulate','rh_rostralmiddlefrontal','rh_superiorfrontal',
'rh_superiorparietal','rh_superiortemporal','rh_supramarginal','rh_frontalpole',
'rh_temporalpole','rh_transversetemporal','rh_insula'
]

assert len(regions) == 68, "Region list should have exactly 68 entries."

# --------------------------------------------------------
# Load first QSM file (determines node count + ordering)
# --------------------------------------------------------
qsm_files = sorted(glob.glob(os.path.join(QSM_NODES_DIR, "*_qsm_nodes.csv")))
if not qsm_files:
    raise FileNotFoundError(f"No QSM node files in {QSM_NODES_DIR}")

first_file = qsm_files[0]
print(f"Using QSM node file: {first_file}")

df = pd.read_csv(first_file)
N = df.shape[0]
print(f"Node count in this file: {N}")

if N != 68:
    raise ValueError(f"Expected 68 rows (1 per region), got {N}")

# --------------------------------------------------------
# Build region label table
# --------------------------------------------------------
region_df = pd.DataFrame({"region": regions})

# --------------------------------------------------------
# Save
# --------------------------------------------------------
os.makedirs(os.path.dirname(OUTFILE), exist_ok=True)
region_df.to_csv(OUTFILE, index=False)

print(f"✔ Saved region labels to: {OUTFILE}")
