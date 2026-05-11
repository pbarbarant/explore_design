# %%
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn.glm.first_level import (
    first_level_from_bids,
    make_first_level_design_matrix,
)
from nilearn.plotting import plot_img_on_surf
from tqdm import tqdm

from exd.events import get_run_events
from exd.models.ideal_observer import SurpriseEstimator


def get_subject_session_runs_beh(data_dir, subject, ses, task_label) -> set:
    """Glob BIDS files to find available run indices for a subject/session."""
    pattern = f"sub-{subject}/ses-{ses}/beh/*task-{task_label}*_run-*_beh.tsv"
    run_files = sorted(Path(data_dir).glob(pattern))
    return set([int(f.stem.split("run-")[1].split("_")[0]) for f in run_files])


def get_fmri_sessions(derivatives_dir, subject, task_label) -> set:
    """Get available fMRI sessions for a subject."""
    pattern = f"sub-{subject}/ses-*/func"
    func_dirs = sorted(Path(derivatives_dir).glob(pattern))
    return set([int(f.parent.name.split("-")[1]) for f in func_dirs])


def make_dmtx_one_run(model, events, confounds):
    estimator = SurpriseEstimator(
        latent_levels=[20, 40, 60, 80],
        sd=events["SD"].unique()[0],
        learning_params={"vol": 0.0416},
        option_cols=["obsA", "obsB"],
        outcome_range=(1, 100),
    )
    pred = estimator.fit_predict(events)
    pred["Surprise"] = pred["Surprise__obsA"] + pred["Surprise__obsB"]
    pred = pred.drop(columns=["Surprise__obsA", "Surprise__obsB"])
    # Concatenate the models_events columns except for onset and duration
    confounds = pd.concat([pred, confounds], axis=1)

    # Create design matrix
    # Interpolate missing values in confounds
    regs = confounds.fillna(0).to_numpy()
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
        t_r=1.71,
        hrf_model="spm + derivative",
        derivatives_folder=derivatives_folder,
        confounds_strategy=["motion"],
        n_jobs=10,
        verbose=10,
    )
    model = models[0]
    run_imgs = run_imgs[0]
    confounds = confounds[0]

    fmri_sessions = get_fmri_sessions(derivatives_dir, subject, task_label)
    img_map = {
        (
            int(p.split("ses-")[1].split("/")[0]),
            int(p.split("run-")[1].split("_")[0]),
        ): p
        for p in run_imgs
    }
    conf_map = {
        (
            int(p.split("ses-")[1].split("/")[0]),
            int(p.split("run-")[1].split("_")[0]),
        ): c
        for p, c in zip(run_imgs, confounds)
    }

    for ses in fmri_sessions:
        beh_runs = get_subject_session_runs_beh(
            derivatives_dir, subject, ses, task_label
        )
        keys = sorted(k for k in img_map if k[0] == ses and k[1] in beh_runs)

        selected_imgs = [img_map[k] for k in keys]
        selected_confounds = [conf_map[k] for k in keys]
        selected_run_ids = [k[1] for k in keys]

        design_matrices = []
        for i, run_id in enumerate(selected_run_ids):
            events = get_run_events(
                derivatives_dir, sub=subject, ses=ses, run=run_id, onset="RT"
            )
            dmtx = make_dmtx_one_run(model, events, selected_confounds[i])
            design_matrices.append(dmtx)
        # TODO: Concatenate design matrices
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
