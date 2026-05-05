# %%
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn.datasets import fetch_surf_fsaverage
from nilearn.glm.first_level import (
    first_level_from_bids,
    make_first_level_design_matrix,
)
from nilearn.plotting import (
    plot_design_matrix,
    plot_img_on_surf,
)
from tqdm import tqdm

from exd.events import get_run_events
from exd.models.ideal_observer import SurpriseEstimator


def filter_confounds(confounds: pd.DataFrame) -> pd.DataFrame:
    # Keep only the motion confounds
    prefixes = ["trans", "rot", "motion"]
    columns = confounds.columns
    filtered_cols = [
        col for col in columns if any(col.startswith(prefix) for prefix in prefixes)
    ]
    return confounds[filtered_cols]


def get_subject_session_runs_beh(data_dir, subject, ses, task_label):
    """Glob BIDS files to find available run indices for a subject/session."""
    pattern = f"sub-{subject}/ses-{ses}/beh/*task-{task_label}*_run-*_beh.tsv"
    run_files = sorted(Path(data_dir).glob(pattern))
    return set([int(f.stem.split("run-")[1].split("_")[0]) for f in run_files])


def get_subject_session_runs_nii(data_dir, subject, ses, task_label):
    """Glob BIDS files to find available run indices for a subject/session."""
    pattern = f"sub-{subject}/ses-{ses}/func/*task-{task_label}*_run-*.nii.gz"
    run_files = sorted(Path(data_dir).glob(pattern))
    return set([int(f.stem.split("run-")[1].split("_")[0]) for f in run_files])


def get_fmri_sessions(derivatives_dir, subject, task_label):
    """Get available fMRI sessions for a subject."""
    pattern = f"sub-{subject}/ses-*/func"
    func_dirs = sorted(Path(derivatives_dir).glob(pattern))
    return set([int(f.parent.name.split("-")[1]) for f in func_dirs])


def make_dmtx_one_run(model, events, confounds):
    estimator = SurpriseEstimator(
        latent_levels=[20, 40, 60, 80],
        sd=10,
        learning_params={"vol": 0.04},
        option_cols=["obsA", "obsB"],
        outcome_range=(1, 100),
    )
    pred = estimator.fit_predict(events)
    pred["Surprise"] = pred["Surprise__obsA"] + pred["Surprise__obsB"]
    pred = pred.drop(columns=["Surprise__obsA", "Surprise__obsB"])

    # Create design matrix
    confounds = filter_confounds(confounds)
    # Concatenate the models_events columns except for onset and duration
    confounds = pd.concat([pred, confounds], axis=1)
    # Interpolate missing values in confounds
    regs = confounds.interpolate(limit_direction="both").to_numpy()
    # Create the design matrix
    n_scans = len(confounds)
    start_time = model.slice_time_ref * model.t_r
    end_time = (n_scans - 1 + model.slice_time_ref) * model.t_r
    frame_times = np.linspace(start_time, end_time, n_scans)
    dmtx = make_first_level_design_matrix(
        frame_times=frame_times,
        events=events,
        hrf_model=model.hrf_model,
        drift_model=model.drift_model,
        high_pass=model.high_pass,
        drift_order=model.drift_order,
        fir_delays=model.fir_delays,
        add_regs=regs,
        add_reg_names=confounds.columns.tolist(),
        min_onset=model.min_onset,
    )
    # Remove the dummy* columns
    dtmx = dmtx.loc[:, ~dmtx.columns.str.startswith("dummy")]
    return dtmx


derivatives_dir = Path(
    "/home/plbarbarant/nasShare/projects/protocols/ExplorePlus_MeynielPaunovRaglio_2024/derivatives/fmriprep-24.1.1_mne-bids-pipeline-1.9.0"
)
data_dir = (
    "/home/plbarbarant/nasShare/projects/protocols/ExplorePlus_MeynielPaunovRaglio_2024"
)
task_label = "ExplorePlus"
space_label = "MNI152NLin2009cAsym"
derivatives_folder = "derivatives/fmriprep-24.1.1_mne-bids-pipeline-1.9.0/"

SUBJECTS = [
    "01",
    "04",
    "05",
    "06",
    "08",
    "09",
    "10",
    "11",
    "13",
    "14",
    "15",
    "16",
    "17",
    "18",
    "19",
    "20",
]


fsaverage = fetch_surf_fsaverage()
fsaverage_infl = {
    "infl_left": fsaverage["infl_left"],
    "infl_right": fsaverage["infl_right"],
}

for subject in tqdm(SUBJECTS, desc="Processing subject: "):
    (
        models,
        run_imgs,
        _,
        confounds,
    ) = first_level_from_bids(
        data_dir,
        task_label,
        space_label,
        sub_labels=[subject],
        smoothing_fwhm=6.0,
        high_pass=1 / 128,
        hrf_model="spm + derivative",
        derivatives_folder=derivatives_folder,
        n_jobs=10,
        verbose=10,
    )
    model = models[0]
    run_imgs = run_imgs[0]
    confounds = confounds[0]

    fmri_sessions = get_fmri_sessions(derivatives_dir, subject, task_label)

    for ses in fmri_sessions:
        run_ids = set.intersection(
            get_subject_session_runs_nii(derivatives_dir, subject, ses, task_label),
            get_subject_session_runs_beh(derivatives_dir, subject, ses, task_label),
        )
        selected_run_idx = [
            i
            for i, img in enumerate(run_imgs)
            if f"ses-{ses}" in img and any(f"run-{i:02d}" in img for i in run_ids)
        ]
        selected_imgs = [run_imgs[i] for i in selected_run_idx]
        selected_confounds = [confounds[i] for i in selected_run_idx]

        design_matrices = []
        for i, run_id in enumerate(run_ids):
            events = get_run_events(derivatives_dir, sub=subject, ses=ses, run=run_id)
            dmtx = make_dmtx_one_run(model, events, selected_confounds[i])
            design_matrices.append(dmtx)
            plot_design_matrix(
                dmtx,
                rescale=True,
                output_file=f"/home/plbarbarant/repos/explore_design/outputs/sub-{subject}_ses-{ses}_run-{run_id}_dmtx.png",
            )

        model.fit(run_imgs=selected_imgs, design_matrices=design_matrices)
        z_map = model.compute_contrast("Surprise", output_type="z_score")

        # Plot positive (left) and negative maps (right)
        plot_img_on_surf(
            stat_map=z_map,
            colorbar=True,
            bg_on_data=False,
            inflate=True,
            title=f"Surprise (z-score) - sub-{subject} ses-{ses}",
            output_file=f"/home/plbarbarant/repos/explore_design/outputs/sub-{subject}_ses-{ses}_surprise.png",
        )
