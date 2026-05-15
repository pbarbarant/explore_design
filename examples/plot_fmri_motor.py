# %%
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn.glm.first_level import (
    first_level_from_bids,
    make_first_level_design_matrix,
)
from nilearn.plotting import (
    plot_design_matrix,
)
from sklearn.base import clone
from tqdm import tqdm

from exd.events import get_run_events
from exd.models.motor import MotorEstimator


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


def make_dmtx_one_run(model, events, confounds, estimator) -> pd.DataFrame:
    algo = clone(estimator)
    events["trial_type"] = algo.fit_predict(events)
    # Create design matrix
    # Replace missing values in confounds
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


def first_level_analysis(
    data_dir,
    derivatives_folder,
    task_label,
    space_label,
    subjects,
    estimator,
):
    (
        models_all_subjects,
        run_imgs_all_subjects,
        _,
        confounds_all_subjects,
    ) = first_level_from_bids(
        data_dir,
        task_label,
        space_label,
        sub_labels=subjects,
        smoothing_fwhm=6.0,
        high_pass=1 / 128,
        t_r=1.71,
        hrf_model="spm + derivative",
        derivatives_folder=derivatives_folder,
        confounds_strategy=["motion", "global_signal"],
        n_jobs=10,
        verbose=10,
    )
    for i, subject in tqdm(enumerate(subjects), desc="Processing subject: "):
        model = models_all_subjects[i]
        run_imgs = run_imgs_all_subjects[i]
        confounds = confounds_all_subjects[i]

        fmri_sessions = get_fmri_sessions(derivatives_dir, subject, task_label)
        run_map = {
            (
                int(p.split("ses-")[1].split("/")[0]),
                int(p.split("run-")[1].split("_")[0]),
            ): (p, c)
            for p, c in zip(run_imgs, confounds)
        }

        for ses in fmri_sessions:
            beh_runs = get_subject_session_runs_beh(
                derivatives_dir, subject, ses, task_label
            )
            # Keep only ses, runs with behavioral data attached
            keys = sorted(k for k in run_map if k[0] == ses and k[1] in beh_runs)

            selected_imgs = [run_map[k][0] for k in keys]
            selected_confounds = [run_map[k][1] for k in keys]
            selected_run_ids = [k[1] for k in keys]

            design_matrices = []
            for run_id, confounds in zip(selected_run_ids, selected_confounds):
                run_events = get_run_events(
                    derivatives_dir, sub=subject, ses=ses, run=run_id, onset="RT"
                )
                dmtx = make_dmtx_one_run(model, run_events, confounds, estimator)
                design_matrices.append(dmtx)
                plot_design_matrix(
                    dmtx,
                    rescale=True,
                    output_file=f"/home/plbarbarant/repos/explore_design/outputs/sub-{subject}_ses-{ses}_run-{run_id}_dmtx_motor.png",
                )

            model.fit(run_imgs=selected_imgs, design_matrices=design_matrices)
            beta_map = model.compute_contrast("R - L", output_type="effect_size")
            beta_map.to_filename(
                f"/home/plbarbarant/repos/explore_design/outputs/sub-{subject}_ses-{ses}_motor.nii.gz",
            )


if __name__ == "__main__":
    derivatives_dir = Path(
        "/home/plbarbarant/nasShare/projects/protocols/ExplorePlus_MeynielPaunovRaglio_2024/derivatives/fmriprep-24.1.1_mne-bids-pipeline-1.9.0"
    )
    data_dir = "/home/plbarbarant/nasShare/projects/protocols/ExplorePlus_MeynielPaunovRaglio_2024"
    task_label = "ExplorePlus"
    space_label = "MNI152NLin2009cAsym"
    derivatives_folder = "derivatives/fmriprep-24.1.1_mne-bids-pipeline-1.9.0/"
    subjects = [
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
    estimator = MotorEstimator(
        option_cols=["obsA", "obsB"],
    )
    first_level_analysis(
        data_dir=data_dir,
        task_label=task_label,
        space_label=space_label,
        derivatives_folder=derivatives_folder,
        subjects=subjects,
        estimator=estimator,
    )
