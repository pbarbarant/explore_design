# %%
from pathlib import Path

import matplotlib.pyplot as plt
from nilearn import plotting

path = Path("/home/plbarbarant/repos/explore_design/outputs/")

SUBJECTS = [
    "01",
    "04",
    "05",
    "06",
    "08",
    "09",
    "10",
    # "11",
    # "13",
    # "14",
    # "15",
    # "16",
    # "17",
    # "18",
    # "19",
    # "20",
]


def concat_img_files(path, qty):
    pattern = f"sub-*_ses-*_{qty}.nii.gz"
    img_files = sorted(list(path.glob(pattern)))
    return img_files


img_files = concat_img_files(path, "motor")
fig, axes = plt.subplots(len(SUBJECTS), 2, figsize=(20, 5 * len(SUBJECTS)))

for subject, ax in zip(SUBJECTS, axes):
    print(f"Processing subject {subject}")
    # Concatenate the two sessions for the current subject
    subject_img_files = [str(f) for f in img_files if f"sub-{subject}" in f.name]
    ses = [f.split("_")[2] for f in subject_img_files]

    plotting.plot_glass_brain(
        subject_img_files[0], axes=ax[0], title=f"Motor map sub-{subject} {ses[0]}"
    )
    plotting.plot_glass_brain(
        subject_img_files[1], axes=ax[1], title=f"Motor map sub-{subject} {ses[1]}"
    )

plt.savefig(path / "concatenated_motor.png", dpi=300, bbox_inches="tight")
plt.show()
