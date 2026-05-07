# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pre_commit

path = Path("/home/plbarbarant/repos/explore_design/outputs/")

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
    # "15",
    "16",
    # "17",
    # "18",
    # "19",
    # "20",
]


def concat_img_files(path, qty):
    pattern = f"sub-*_ses-*_{qty}.png"
    img_files = sorted(list(path.glob(pattern)))
    img_files = [f for f in img_files if "dmtx" not in str(f)]
    return img_files


img_files = concat_img_files(path, "motor")

total_array = []
for subject in SUBJECTS:
    print(f"Processing subject {subject}")
    # Concatenate the two sessions for the current subject
    subject_img_files = [str(f) for f in img_files if f"sub-{subject}" in f.name]

    array1 = plt.imread(subject_img_files[0])
    array2 = plt.imread(subject_img_files[1])
    concatenated_array = np.concatenate((array1, array2), axis=1)

    total_array.append(concatenated_array)

total_array = np.concatenate(total_array, axis=0)

plt.imshow(total_array)
# remove the axes
plt.axis("off")
plt.savefig(path / "concatenated_motor.png", dpi=2000, bbox_inches="tight")
plt.show()
