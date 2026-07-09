# %%
from pathlib import Path

import pandas as pd
from nilearn.datasets import load_fsaverage, load_fsaverage_data
from nilearn.glm import threshold_stats_img
from nilearn.glm.first_level.first_level import _list_valid_subjects
from nilearn.glm.second_level import SecondLevelModel, make_second_level_design_matrix
from nilearn.plotting import plot_design_matrix, plot_surf_stat_map, show
from nilearn.surface import SurfaceImage

data_dir = Path(
    "/home/plbarbarant/nasShare/projects/protocols/ExplorePlus_MeynielPaunovRaglio_2024"
)
task_label = "ExplorePlus"
space_label = "MNI152NLin2009cAsym"
derivatives_folder = "derivatives/fmriprep-24.1.1_mne-bids-pipeline-1.9.0/"
subjects_label = _list_valid_subjects(str(data_dir / derivatives_folder), None)
quantity_name = "motor"

imgs_L = sorted(
    list(
        Path("/home/plbarbarant/repos/explore_design/outputs/").rglob(
            f"sub-*/ses-*/quantity_{quantity_name}_stat-effect_size_hemi-L*"
        )
    )
)
imgs_R = sorted(
    list(
        Path("/home/plbarbarant/repos/explore_design/outputs/").rglob(
            f"sub-*/ses-*/quantity_{quantity_name}_stat-effect_size_hemi-R*"
        )
    )
)
imgs = []
subjects_label = []
fsaverage_meshes = load_fsaverage()
for img_L, img_R in zip(imgs_L, imgs_R):
    data = {
        "left": img_L,
        "right": img_R,
    }
    subjects_label.append("sub-" + str(img_L)[51:53])
    imgs.append(SurfaceImage(mesh=fsaverage_meshes["pial"], data=data))

confounds = pd.read_csv(data_dir / "rawdata" / "participants.tsv", sep="\t")[
    ["participant_id", "sex", "hand", "age"]
]
confounds.rename(columns={"participant_id": "subject_label"}, inplace=True)
confounds["sex"] = confounds["sex"].map(lambda s: 0 if s == "M" else 1)
confounds["hand"] = confounds["hand"].map(lambda s: 0 if s == "R" else 1)
confounds["age"] = (confounds["age"] - confounds["age"].mean()) / confounds["age"].std()
dmtx = make_second_level_design_matrix(
    subjects_label=subjects_label,
    confounds=confounds,
)
plot_design_matrix(dmtx)
show()

# %%
model = SecondLevelModel(
    n_jobs=10,
    verbose=10,
)
model.fit(imgs, design_matrix=dmtx)
stat_map = model.compute_contrast("intercept", output_type="z_score")

thresholded_map, threshold = threshold_stats_img(
    stat_map, alpha=0.05, height_control="fdr"
)
fsaverage_sulcal = load_fsaverage_data(data_type="sulcal", mesh_type="pial")
# %%
# Plot your second-level stat map (left hemi example)
# from nilearn import datasets, surface
import matplotlib.pyplot as plt
import numpy as np
from nilearn import datasets
from nilearn.plotting import plot_surf_contours

# Load Destrieux atlas (fsaverage parcellation)
destrieux = datasets.fetch_atlas_surf_destrieux()
destrieux_atlas = SurfaceImage(
    mesh=fsaverage_meshes["inflated"],
    data={
        "left": destrieux["map_left"],
        "right": destrieux["map_right"],
    },
)

# Destrieux labels relevant to motor cortex (precentral gyrus / sulcus etc.)
# Check exact label names for your version:
regions_dict = {
    "G_postcentral": "Postcentral gyrus",
    "S_central": "Central sulcus",
    "G_precentral": "Precentral gyrus",
}
regions_indices = [
    np.where(np.array(destrieux["labels"]) == region)[0][0] for region in regions_dict
]
labels = list(regions_dict.values())

# %%

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")
plot_surf_stat_map(
    fsaverage_meshes["inflated"],
    stat_map=thresholded_map,
    hemi="both",
    bg_map=fsaverage_sulcal,
    bg_on_data=True,
    threshold=threshold,
    cmap="cold_hot",
    axes=ax,
    view="dorsal",
    title=f"Second Level z-score FDR .05 - {quantity_name}",
)
plot_surf_contours(
    roi_map=destrieux_atlas,
    labels=labels,
    hemi="both",
    levels=regions_indices,  # only draw contours for motor regions
    figure=fig,
    axes=ax,
    legend=True,
    colors=["r", "g", "b"],
)

fig.show()
