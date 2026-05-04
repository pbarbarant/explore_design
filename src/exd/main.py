# %%
import numpy as np
import pandas as pd
from nilearn.glm.first_level import (
    first_level_from_bids,
    make_first_level_design_matrix,
)
from nilearn.plotting import (
    plot_design_matrix,
    plot_event,
    plot_stat_map,
    show,
)

from exd.models.motor import MotorEstimator


def make_events(features: pd.DataFrame):
    onset = features.trial_start / 1000
    duration = (features.trial_end - features.trial_start) / 1000
    features = features.drop(columns=["Unnamed: 0"])
    events = pd.concat(
        [onset.rename("onset"), duration.rename("duration"), features],
        axis=1,
    )
    return events


def get_run_events(ses, run):
    path_beh = f"/home/plbarbarant/nasShare/projects/protocols/ExplorePlus_MeynielPaunovRaglio_2024/derivatives/fmriprep-24.1.1_mne-bids-pipeline-1.9.0/sub-01/ses-{ses}/beh/sub-01_ses-{ses}_task-ExplorePlus_run-{run:02d}_desc-formatted_beh.tsv"
    features = pd.read_csv(path_beh, sep="\t", index_col=None)
    events = make_events(features)
    return events


# estimator = DeltaRuleEstimator(
#     learning_params={"lr-cu.c": 0.04, "lr-cu.u": 0.00, "lr-sd.low": 0.01},
#     initial_prediction=50,
#     option_cols=["obsA", "obsB"],
#     condition_cols=["forced", "SD"],
# )

models_events = []
for i in range(1, 5):
    events = get_run_events(1, i)
    estimator = MotorEstimator()
    pred = estimator.fit_predict(events)
    events_full = pd.concat([events, pred], axis=1)
    models_events.append(events_full[["onset", "duration", "trial_type"]])
# %%

data_dir = (
    "/home/plbarbarant/nasShare/projects/protocols/ExplorePlus_MeynielPaunovRaglio_2024"
)
task_label = "ExplorePlus"
space_label = "MNI152NLin2009cAsym"
derivatives_folder = "derivatives/fmriprep-24.1.1_mne-bids-pipeline-1.9.0/"
n_runs = 9
(
    models,
    models_run_imgs,
    _,
    models_confounds,
) = first_level_from_bids(
    data_dir,
    task_label,
    space_label,
    sub_labels=["01"],
    smoothing_fwhm=6.0,
    high_pass=1 / 128,
    hrf_model="spm + derivative",
    derivatives_folder=derivatives_folder,
    n_jobs=n_runs,
    verbose=10,
)
# %%
# Plot events matrix
plot_event(models_events, cmap="tab10")
show()


# %%
def filter_confounds(confounds: pd.DataFrame) -> pd.DataFrame:
    # Keep only the motion confounds
    prefixes = ["trans", "rot", "motion"]
    columns = confounds.columns
    filtered_cols = [
        col for col in columns if any(col.startswith(prefix) for prefix in prefixes)
    ]
    return confounds[filtered_cols]


for i in range(len(models_confounds[0])):
    models_confounds[0][i] = filter_confounds(models_confounds[0][i])

# Plot design matrix
run_idx = 3
n_scans = len(models_confounds[0][run_idx])
start_time = models[0].slice_time_ref * models[0].t_r
end_time = (n_scans - 1 + models[0].slice_time_ref) * models[0].t_r
frame_times = np.linspace(start_time, end_time, n_scans)
design = make_first_level_design_matrix(
    frame_times=frame_times,
    events=models_events[run_idx],
    hrf_model=models[0].hrf_model,
    drift_model=models[0].drift_model,
    high_pass=models[0].high_pass,
    drift_order=models[0].drift_order,
    fir_delays=models[0].fir_delays,
    add_regs=models_confounds[0][run_idx]
    .interpolate(limit_direction="both")
    .to_numpy(),
    add_reg_names=models_confounds[0][run_idx].columns.tolist(),
    min_onset=models[0].min_onset,
)
plot_design_matrix(design, rescale=True)
show()
# %%
model_test = models[0].fit(
    run_imgs=models_run_imgs[0][:4],
    events=models_events,
    confounds=[x.interpolate(limit_direction="both") for x in models_confounds[0][:4]],
)

# %%
z_map = model_test.compute_contrast("R - L")

plot_stat_map(z_map, colorbar=True, threshold=1.96, display_mode="z")
show()
# %%

plot_stat_map(model_test.mask_img)
show()
