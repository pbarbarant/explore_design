# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from nilearn.datasets import load_fsaverage, load_fsaverage_data
from nilearn.glm import threshold_stats_img
from nilearn.glm.first_level.first_level import _list_valid_subjects
from nilearn.plotting import plot_surf, plot_surf_stat_map, show
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
            f"sub-*/ses-*/quantity_{quantity_name}_stat-z_score_hemi-L*"
        )
    )
)
imgs_R = sorted(
    list(
        Path("/home/plbarbarant/repos/explore_design/outputs/").rglob(
            f"sub-*/ses-*/quantity_{quantity_name}_stat-z_score_hemi-R*"
        )
    )
)
imgs = []
subjects_label = []
sessions = []
fsaverage_meshes = load_fsaverage()
for img_L, img_R in zip(imgs_L, imgs_R):
    data = {
        "left": img_L,
        "right": img_R,
    }
    subjects_label.append("sub-" + str(img_L)[51:53])
    sessions.append(str(img_L)[54:59])
    imgs.append(SurfaceImage(mesh=fsaverage_meshes["pial"], data=data))


def threshold_stat_map(stat_map_img, alpha=0.05, height_control="fdr"):
    """Threshold a stat map using FDR correction."""
    thresholded_map, threshold = threshold_stats_img(
        stat_map_img, alpha=alpha, height_control=height_control
    )
    return thresholded_map, threshold


# Group images by subject: {subject_label: [img_ses1, img_ses2]}
subject_to_imgs = {}
subject_to_sessions = {}
for sub, ses, img in zip(subjects_label, sessions, imgs):
    subject_to_imgs.setdefault(sub, []).append(img)
    subject_to_sessions.setdefault(sub, []).append(ses)

# Debug: only keep the first two subjects
debug_subjects = list(subject_to_imgs.keys())[:2]

# %%
# For each subject, threshold both sessions and build a red/blue/green
# overlap map per hemisphere: 1 = session 1 only, 2 = session 2 only,
# 3 = overlap. 0 (masked out) = active in neither session.
sulcal_data = load_fsaverage_data(mesh_type="inflated", data_type="sulcal")
views = ["lateral"]  # , "medial"]
hemis = ["left", "right"]

n_subjects = len(debug_subjects)
n_cols = len(hemis) * len(views)
fig, axes = plt.subplots(
    n_subjects,
    n_cols,
    figsize=(4 * n_cols, 4 * n_subjects),
    subplot_kw={"projection": "3d"},
)
if n_subjects == 1:
    axes = axes[None, :]

for row, sub in enumerate(debug_subjects):
    print(f"Processing subject {sub} ({row + 1}/{n_subjects})")
    img_ses1, img_ses2 = subject_to_imgs[sub][-2:]
    ses1, ses2 = subject_to_sessions[sub][-2:]

    thresholded_ses1, threshold1 = threshold_stat_map(img_ses1)
    thresholded_ses2, threshold2 = threshold_stat_map(img_ses2)
    col = 0
    for hemi in hemis:
        for view in views:
            ax = axes[row, col]
            plot_surf(
                fsaverage_meshes["inflated"],
                surf_map=0 * sulcal_data.data.parts[hemi],
                view=view,
                hemi=hemi,
                axes=ax,
                figure=fig,
                bg_map=sulcal_data.data.parts[hemi],
                bg_on_data=True,
                cmap="gray",
                threshold=1,
                colorbar=False,
            )
            plot_surf_stat_map(
                fsaverage_meshes["inflated"],
                thresholded_ses1.data.parts[hemi],
                view=view,
                hemi=hemi,
                axes=ax,
                figure=fig,
                cmap="autumn",
                threshold=threshold1,
                colorbar=False,
                alpha=0.2,
                bg_on_data=True,
            )
            plot_surf_stat_map(
                fsaverage_meshes["inflated"],
                thresholded_ses2.data.parts[hemi],
                view=view,
                hemi=hemi,
                axes=ax,
                figure=fig,
                cmap="RdBu_r",
                threshold=threshold2,
                colorbar=False,
                alpha=0.1,
            )

            if col == 0:
                ax.set_title(
                    f"{sub} - red = {ses1} only | blue = {ses2} only | green = overlap)",
                    fontsize=12,
                )
            col += 1

fig.suptitle(
    f"{quantity_name} - FDR .05",
    fontsize=14,
)
fig.tight_layout()
# fig.savefig(
#     f"/home/plbarbarant/repos/explore_design/outputs/qc/{quantity_name}.png",
#     dpi=100,
#     bbox_inches="tight",
# )
show()
