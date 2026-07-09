# %%
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn.glm.first_level import compute_regressor, make_first_level_design_matrix
from nilearn.glm.first_level.first_level import _list_valid_subjects

from exd.fmri_utils import first_level_analysis
from exd.models.ideal_observer import IdealObserverEstimator


def make_dmtx_one_run_Er(model, events, confounds) -> pd.DataFrame:
    estimator = IdealObserverEstimator(
        latent_levels=[20, 40, 60, 80],
        sd=events["SD"].unique()[0],
        learning_params={"vol": 0.0416},
        option_cols=["obsA", "obsB"],
        outcome_range=(1, 100),
    )
    events = pd.concat([events, estimator.fit_predict(events)], axis=1)
    diff = events["IO__obsA"] - events["IO__obsB"]
    sign = np.where(events["arm_choice"].eq("A"), 1, -1)

    events["Er"] = diff * sign

    # Frame times
    n_scans = len(confounds)
    start_time = model.slice_time_ref * model.t_r
    end_time = (n_scans - 1 + model.slice_time_ref) * model.t_r
    frame_times = np.linspace(start_time, end_time, n_scans)

    # Er events without modulation
    condition = np.array(
        [
            events["onset"].values,
            events["duration"].values,
            np.ones(len(events)),  # amplitude = 1, i.e. unmodulated
        ]
    )
    er_no_mod, _ = compute_regressor(
        exp_condition=condition,
        hrf_model=model.hrf_model,
        frame_times=frame_times,
        min_onset=model.min_onset,
    )
    confounds = confounds.copy()
    confounds.insert(0, "Er_no_modulation", er_no_mod[:, 0])

    # Er events with modulation
    modulated_events = pd.DataFrame(
        {
            "onset": events["onset"].values,
            "duration": events["duration"].values,
            "trial_type": "Er",
            "modulation": events["Er"].values,
        }
    )

    # Replace missing values in confounds
    regs = confounds.fillna(0).to_numpy()

    dmtx = make_first_level_design_matrix(
        frame_times=frame_times,
        events=modulated_events,
        hrf_model=model.hrf_model,
        drift_model=model.drift_model,
        high_pass=model.high_pass,
        drift_order=model.drift_order,
        fir_delays=model.fir_delays,
        add_regs=regs,
        add_reg_names=confounds.columns.tolist(),
        min_onset=model.min_onset,
    )

    # Remove dummy columns
    dmtx = dmtx.loc[:, ~dmtx.columns.str.startswith("dummy")]
    return dmtx


if __name__ == "__main__":
    data_dir = Path(
        "/home/plbarbarant/nasShare/projects/protocols/ExplorePlus_MeynielPaunovRaglio_2024"
    )
    task_label = "ExplorePlus"
    derivatives_folder = "derivatives/fmriprep-24.1.1_mne-bids-pipeline-1.9.0/"
    subjects = _list_valid_subjects(str(data_dir / derivatives_folder), None)
    first_level_analysis(
        data_dir=str(data_dir),
        task_label=task_label,
        derivatives_folder=derivatives_folder,
        subjects=subjects,
        quantity_name="Er",
        onset="cue",
        contrast_name="Er",
        dmtx_functor=make_dmtx_one_run_Er,
    )
