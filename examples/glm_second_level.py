# %%
from pathlib import Path

import pandas as pd
from nilearn.datasets import load_fsaverage
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
mesh = load_fsaverage("fsaverage5")["pial"]
for img_L, img_R in zip(imgs_L, imgs_R):
    data = {
        "left": img_L,
        "right": img_R,
    }
    subjects_label.append("sub-" + str(img_L)[51:53])
    imgs.append(SurfaceImage(mesh=mesh, data=data))

# %%
confounds = pd.read_csv(data_dir / "rawdata" / "participants.tsv", sep="\t")[
    ["participant_id", "sex"]
]
confounds.rename(columns={"participant_id": "subject_label"}, inplace=True)
confounds["sex"] = confounds["sex"].map(lambda s: 0 if s == "M" else 1)
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
# %%
plot_surf_stat_map(
    stat_map=stat_map,
    hemi="both",
    # threshold=1.96,
    cmap="cold_hot",
    bg_on_data=False,
    title=f"Second Level z-score - {quantity_name}",
)
show()
