# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from nilearn.datasets import load_fsaverage, load_fsaverage_data
from nilearn.glm import threshold_stats_img
from nilearn.glm.first_level.first_level import _list_valid_subjects
from nilearn.plotting import show
from nilearn.surface import SurfaceImage
from nilearn.surface import load_surf_mesh
from nilearn.plotting.cm import mix_colormaps
from nilearn.plotting.surface._matplotlib_backend import (
    _compute_facecolors,
    _get_view_plot_surf,
)
import matplotlib.colors as mcolors
import matplotlib.cm as cm

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
debug_subjects = list(subject_to_imgs.keys())

# %%
# For each subject, threshold both sessions and build a red/blue/green
# overlap map per hemisphere: 1 = session 1 only, 2 = session 2 only,
# 3 = overlap. 0 (masked out) = active in neither session.
sulcal_data = load_fsaverage_data(mesh_type="inflated", data_type="sulcal")
views = ["lateral" , "medial"]
hemis = ["left", "right"]
alpha = 0.05

n_subjects = len(debug_subjects)
n_cols = len(hemis) * len(views)
fig, axes = plt.subplots(
    n_subjects,
    n_cols,
    figsize=(4 * n_cols, 4 * n_subjects),
    subplot_kw={"projection": "3d"},
)

def _plot_surf(
    surf_mesh,
    axes,
    figure,
    surf_map=None,
    bg_map=None,
    hemi="left",
    view=None,
    avg_method=None,
    alpha=None,
    title=None,
):
    """
    Adapted from 'nilearn/plotting/surface/_matplotlib_backend.py'

    Implement 'matplotlib' backend code for
    `~nilearn.plotting.surface.surf_plotting.plot_surf` function.

    Parameters
    ----------
    surf_map : 2D array (n_vertices, 4): RGBA colors per vertex (values 0-1)
    """
    # adjust non-common params
    if avg_method is None:
        avg_method = "mean"
    if alpha is None:
        alpha = "auto"

    coords, faces = load_surf_mesh(surf_mesh)

    # Center the mesh
    coords -= np.mean(coords, axis=0)
    limits = [coords.min(), coords.max()]

    # Get elevation and azimut from view
    elev, azim = _get_view_plot_surf(hemi, view)

    axes.set_xlim(*limits)
    axes.set_ylim(*limits)
    axes.view_init(elev=elev, azim=azim)
    axes.set_axis_off()

    # plot mesh without data
    p3dcollec = axes.plot_trisurf(
        coords[:, 0],
        coords[:, 1],
        coords[:, 2],
        triangles=faces,
        linewidth=0.1,
        antialiased=False,
        color="white",
    )

    # reduce viewing distance to remove space around mesh
    axes.set_box_aspect(None, zoom=1.3)

    # Compute background face colors
    bg_face_colors = _compute_facecolors(bg_map, faces, coords.shape[0], alpha)

    if surf_map is not None:
        # Check if surf_map contains RGB(A) colors
        surf_map = np.asarray(surf_map)

        # RGB or RGBA colors provided per vertex
        # Average vertex colors to get face colors
        face_colors = np.mean(surf_map[faces], axis=1)

        # Clip values to valid range [0, 1]
        face_colors = np.clip(face_colors, 0, 1)

        # Mix with background colors
        face_colors = mix_colormaps(face_colors, bg_face_colors)

        p3dcollec.set_facecolors(face_colors)
        p3dcollec.set_edgecolors(face_colors)
    else:
        # No surf_map, use background colors
        p3dcollec.set_facecolors(bg_face_colors)
        p3dcollec.set_edgecolors(bg_face_colors)

    if title is not None:
        axes.set_title(title)

    return figure

def add_row_colorbars(fig, row_axes, cmap1, cmap2, norm, label1="ses1", label2="ses2"):
    """
    row_axes: list of Axes objects belonging to one row (e.g. all hemi/view panels for one subject)
    """
    # get bounding box spanning all axes in this row, in figure coordinates
    positions = [ax.get_position() for ax in row_axes]
    left = min(p.x0 for p in positions)
    right = max(p.x1 for p in positions)
    bottom = min(p.y0 for p in positions)

    cbar_height = 0.012
    cbar_gap = 0.015
    cbar_y = bottom - cbar_gap - cbar_height  # just under the row

    row_width = right - left
    cbar_width = row_width * 0.35
    gap_between = row_width * 0.06

    cbar_ax1 = fig.add_axes([left + row_width/2 - gap_between/2 - cbar_width, cbar_y, cbar_width, cbar_height])
    cbar_ax2 = fig.add_axes([left + row_width/2 + gap_between/2, cbar_y, cbar_width, cbar_height])

    sm1 = plt.cm.ScalarMappable(cmap=cmap1, norm=norm)
    sm1.set_array([])
    cb1 = fig.colorbar(sm1, cax=cbar_ax1, orientation="horizontal")
    cb1.set_label(label1, fontsize=8)
    cb1.ax.tick_params(labelsize=7)

    sm2 = plt.cm.ScalarMappable(cmap=cmap2, norm=norm)
    sm2.set_array([])
    cb2 = fig.colorbar(sm2, cax=cbar_ax2, orientation="horizontal")
    cb2.set_label(label2, fontsize=8)
    cb2.ax.tick_params(labelsize=7)

for row, sub in enumerate(debug_subjects):
    print(f"Processing subject {sub} ({row + 1}/{n_subjects})")
    img_ses1, img_ses2 = subject_to_imgs[sub][-2:]
    ses1, ses2 = subject_to_sessions[sub][-2:]

    thresholded_ses1, threshold1 = threshold_stat_map(img_ses1, alpha=alpha)
    thresholded_ses2, threshold2 = threshold_stat_map(img_ses2, alpha=alpha)
    col = 0
    for hemi in hemis:
        row_axes = []
        for view in views:
            ax = axes[row, col]
            val1 = thresholded_ses1.data.parts[hemi]
            val2 = thresholded_ses2.data.parts[hemi]

            mask1 = val1 != 0
            mask2 = val2 != 0
            overlap = mask1 & mask2

            # symmetric normalization: same vmax magnitude on both sides of 0
            vmax = np.nanmax(np.abs(np.concatenate([val1, val2])))

            norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

            cmap1 = cm.get_cmap("RdBu_r")   # ses1
            cmap2 = cm.get_cmap("cold_hot") # ses2

            rgb1 = cmap1(norm(val1))[:, :3]
            rgb2 = cmap2(norm(val2))[:, :3]

            n = 10242
            rgb = np.zeros((n, 3))
            rgb[mask1 & ~mask2] = rgb1[mask1 & ~mask2]
            rgb[mask2 & ~mask1] = rgb2[mask2 & ~mask1]
            # for overlap, blend the two session colors (or pick a fixed highlight color instead)
            rgb[overlap] = (rgb1[overlap] + rgb2[overlap]) / 2

            alphas = (mask1 | mask2).astype(float)

            surf_map = np.column_stack([rgb, alphas])
            surf_map = np.nan_to_num(surf_map)

            _plot_surf(
                fsaverage_meshes["inflated"].parts[hemi],
                surf_map=surf_map,
                view=view,
                hemi=hemi,
                axes=ax,
                bg_map=sulcal_data.data.parts[hemi],
                figure=fig,
            )
            col += 1
            row_axes.append(ax)

        add_row_colorbars(
            fig, row_axes,
            cmap1=cmap1, cmap2=cmap2, norm=norm,
            label1=f"{sub} {ses1}", label2=f"{sub} {ses2}"
        )
fig.suptitle(
    f"{quantity_name} - FDR {alpha}",
    fontsize=14,
)

fig.savefig(
    f"/home/plbarbarant/repos/explore_design/outputs/qc/{quantity_name}.png",
    dpi=100,
    bbox_inches="tight",
)
show()
