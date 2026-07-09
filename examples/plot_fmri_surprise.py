# %%
from pathlib import Path

from nilearn.glm.first_level.first_level import _list_valid_subjects

from exd.fmri_utils import first_level_analysis, make_modulated_dtmx
from exd.models.ideal_observer import SurpriseEstimator


def make_dmtx_one_run_surprise(model, events, confounds):
    estimator = SurpriseEstimator(
        latent_levels=[20, 40, 60, 80],
        sd=events["SD"].unique()[0],
        learning_params={"vol": 0.0416},
        option_cols=["obsA", "obsB"],
        outcome_range=(1, 100),
    )
    pred = estimator.fit_predict(events)
    events["Surprise"] = pred["Surprise__obsA"] + pred["Surprise__obsB"]

    return make_modulated_dtmx(
        model=model, events=events, confounds=confounds, quantity_name="Surprise"
    )


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
        quantity_name="surprise",
        onset="outcome",
        contrast_name="Surprise",
        dmtx_functor=make_dmtx_one_run_surprise,
    )
