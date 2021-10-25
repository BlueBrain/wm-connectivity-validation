import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from voxcell import RegionMap


def load_brain_atlas(c, regions):
    """
    Load circuit c's brain atlas and returns atlas regions per voxel, layers, region names,
    atlas ids for given layer/region, rel. depth per voxel, and rel. layer depth boundaries
    per layer/region
    """
    # Load brain atlas
    region_map = RegionMap.load_json(c.atlas.fetch_hierarchy())
    atlas_regions = c.atlas.load_data('brain_regions')

    rids = list(np.unique(atlas_regions.raw[atlas_regions.raw > 0])) # Region ids
    racrs = [region_map.get(rid, 'acronym') for rid in rids] # Region acronyms
    layers = list(np.unique([int(racr.split(';')[1][1:]) for racr in racrs]))

    atlas_ids = np.zeros((len(layers), len(regions)), dtype=int) # Atlas ids for given layers/regions
    for ridx, reg in enumerate(regions):
        for lidx, lay in enumerate(layers):
            atlas_ids[lidx, ridx] = list(region_map.find(f'{reg};L{lay}', 'acronym'))[0]

    # Compute voxel-based depth [depth = height - distance] and rel. depth [rel_depth = (height - distance) / height] values
    atlas_height = c.atlas.load_data('height')
    atlas_distance = c.atlas.load_data('distance')
    vox_depth = atlas_height.raw - atlas_distance.raw
    vox_rel_depth = (atlas_height.raw - atlas_distance.raw) / atlas_height.raw

    print(f'LOADED {len(layers)} layers and {len(regions)} regions from brain atlas')

    return atlas_regions, layers, atlas_ids, vox_rel_depth


def estimate_layer_boundaries(vox_rel_depth, atlas_regions, atlas_ids):
    """
    Estimate rel. layer depth boundaries for given list of region
    containing numpy arrays of atlas ids
    """
    if not isinstance(atlas_ids, list):
        atlas_ids = [atlas_ids]
    num_regions = len(atlas_ids)

    # Rel. depth range per layer
    rel_layer_depth_range_raw = []
    for ridx in range(num_regions):
        num_layers = atlas_ids[ridx].shape[0]
        depth_range = np.zeros((num_layers, 2))
        for lidx in range(num_layers):
            dval_tmp = vox_rel_depth.flatten()[np.in1d(atlas_regions.raw.flatten(), atlas_ids[ridx][lidx, :].flatten())]
            depth_range[lidx, :] = [np.nanmin(dval_tmp), np.nanmax(dval_tmp)]
        rel_layer_depth_range_raw.append(depth_range)

    rel_layer_depth_range = []
    rel_layer_thickness = []
    for depth_range in rel_layer_depth_range_raw:
        # Estimate mid-points between layers
        mid_points = np.mean(np.reshape(depth_range.flatten()[1:-1], (depth_range.shape[0] - 1, -1)), 1)
        mid_points = np.concatenate(([depth_range[0, 0]], mid_points, [depth_range[-1, -1]]))
        rel_layer_thickness.append(np.diff(mid_points))

        # Re-define rel. depth range per layer (so that continuous range between consecutive layers)
        depth_range = np.concatenate((mid_points[[0]], mid_points[np.repeat(np.arange(1, len(mid_points) - 1), 2)], mid_points[[-1]]))
        depth_range = np.reshape(depth_range, (num_layers, -1))
        rel_layer_depth_range.append(depth_range)

    return rel_layer_depth_range, rel_layer_thickness


def estimate_depth_profiles(proj_file, atlas_regions, atlas_ids, vox_rel_depth, num_rel_depth_bins=50):
    """
    Estimate rel. depth profiles of voxel-based synapse densities of given midrange projection
    for given list of regions containing numpy arrays of atlas ids
    """
    if not isinstance(atlas_ids, list):
        atlas_ids = [atlas_ids]
    num_regions = len(atlas_ids)
    print(f'ESTIMATING DEPTH PROFILES for {num_regions} (combined) region selection(s) of projection {os.path.split(proj_file)[1]} ...', end=' ')

    # Load synapse property tables (incl. positions) of midrange projections
    syn_table = pd.read_feather(proj_file)

    # Voxel-based synapse densities
    syn_pos = syn_table[['x', 'y', 'z']]
    syn_atlas_idx = atlas_regions.positions_to_indices(syn_pos.values)
    idx, cnt = np.unique(syn_atlas_idx, axis=0, return_counts=True)
    vox_syn_count = np.zeros_like(atlas_regions.raw, dtype=int)
    vox_syn_count[idx[:, 0], idx[:, 1], idx[:, 2]] += cnt # Count synapses per voxel => MULTIPLE OCCURRENCES TO BE TAKEN INTO ACCOUNT!!
    vox_syn_density = vox_syn_count / atlas_regions.voxel_volume # (#Syn/um3)
    
    # Depth histogram per region
    rel_depth_bins = np.linspace(np.nanmin(vox_rel_depth), np.nanmax(vox_rel_depth), num_rel_depth_bins + 1)

    rel_depth_values = vox_rel_depth[~np.isnan(vox_rel_depth)]
    density_values = vox_syn_density[~np.isnan(vox_rel_depth)]
    regid_values = atlas_regions.raw[~np.isnan(vox_rel_depth)]

    rel_depth_density_hist = np.zeros((num_rel_depth_bins, num_regions))
    for ridx in range(num_regions):
        print(f'{ridx + 1}', end=' ' if ridx < num_regions - 1 else '\n')
        for didx in range(num_rel_depth_bins):
            dmin = rel_depth_bins[didx]
            dmax = rel_depth_bins[didx + 1]
            if didx + 1 == num_rel_depth_bins:
                dmax += 1 # So that border values also included in last bin
            dsel = np.logical_and(rel_depth_values >= dmin, rel_depth_values < dmax)
            rsel = np.in1d(regid_values, atlas_ids[ridx].flatten())
            if np.sum(np.logical_and(dsel, rsel)) > 0:
                rel_depth_density_hist[didx, ridx] = np.mean(density_values[np.logical_and(dsel, rsel)]) # Mean density in a given depth range
            else:
                rel_depth_density_hist[didx, ridx] = 0.0

    return rel_depth_density_hist, rel_depth_bins


def rel_density_profile_from_recipe(recipe, profile_name):
    """
    Extract rel. density profile from recipe
    """
    profile_idx = np.where([profile['name'] == profile_name for profile in recipe['layer_profiles']])[0]
    if len(profile_idx) == 0:
        print(f'ERROR: Profile {profile_name} not found!')
        return []
    profile_idx = profile_idx[0]
    profile_recipe = recipe['layer_profiles'][profile_idx]['relative_densities']

    num_layers = np.max([np.max([int(lay[1:]) for lay in p_dict['layers']]) for p_dict in profile_recipe])
    rel_density_layer_profile = np.zeros(num_layers)
    for p_idx, p_dict in enumerate(profile_recipe):
        for lay in p_dict['layers']:
            rel_density_layer_profile[int(lay[1:]) - 1] = p_dict['value']

    return rel_density_layer_profile


def plot_rel_density_profiles(rel_depth_density_hist, rel_depth_bins, regions, rel_layer_depth_range, density_layer_profile, err_bars=None, fig_title=None, unit=None, num_rows=1, save_path=None):
    """
    Plot rel. synapse density profiles (voxel-based)
    """
    if err_bars is not None:
        assert err_bars.shape == rel_depth_density_hist.shape, 'ERROR: Error bar shape mismatch!'
        max_range = 1.05 * np.maximum(np.max(rel_depth_density_hist + err_bars), np.max(density_layer_profile))
    else:
        max_range = 1.05 * np.maximum(np.max(rel_depth_density_hist), np.max(density_layer_profile))
    num_rel_depth_bins = len(rel_depth_bins) - 1
    rel_depth_bin_centers = np.array([np.mean(rel_depth_bins[i : i + 2]) for i in range(num_rel_depth_bins)])
    num_cols = np.ceil(len(regions) / num_rows).astype(int)
    plt.figure(figsize=(2 * num_cols, 3 * num_rows))
    plt.gcf().patch.set_facecolor('w')
    for ridx, reg in enumerate(regions):
        plt.subplot(num_rows, num_cols, ridx + 1)
        plt.barh(100.0 * rel_depth_bin_centers, rel_depth_density_hist[:, ridx], np.diff(100.0 * rel_depth_bin_centers[:2]))
        if err_bars is not None:
            plt.errorbar(rel_depth_density_hist[:, ridx], 100.0 * rel_depth_bin_centers, xerr=err_bars[:, ridx], fmt='|', markersize=1.0, markeredgewidth=0.5, color='k', elinewidth=0.5)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.xlim([0, max_range])
        plt.ylim([100.0 * rel_depth_bins[0], 100.0 * rel_depth_bins[-1]])
        plt.gca().invert_yaxis()
        plt.xticks(rotation=45)
        plt.title(reg)
        if unit:
            plt.xlabel(unit)

        ## Plot layer boundaries
        num_layers = rel_layer_depth_range[0].shape[0]
        lcolors = plt.cm.jet(np.linspace(0, 1, num_layers))
        for lidx in range(num_layers):
            plt.plot(np.ones(2) * max(plt.xlim()), 100.0 * rel_layer_depth_range[ridx][lidx, :], '-_', color=lcolors[lidx, :], linewidth=5, alpha=0.5, solid_capstyle='butt', markersize=10, clip_on=False)
            plt.text(max(plt.xlim()), 100.0 * np.mean(rel_layer_depth_range[ridx][lidx, :]), f'  L{lidx + 1}', color=lcolors[lidx, :], ha='left', va='center')
            plt.plot(plt.xlim(), np.ones(2) * 100.0 * rel_layer_depth_range[ridx][lidx, 0], '-', color=lcolors[lidx, :], linewidth=1, alpha=0.1, zorder=0)
            plt.plot(plt.xlim(), np.ones(2) * 100.0 * rel_layer_depth_range[ridx][lidx, 1], '-', color=lcolors[lidx, :], linewidth=1, alpha=0.1, zorder=0)
        if np.mod(ridx, np.ceil(len(regions) / num_rows).astype(int)) == 0:
            plt.ylabel('Rel. depth [%]')
        else:
            plt.gca().set_yticklabels([])

        plt.step(np.repeat(density_layer_profile[ridx], 2), 100.0 * rel_layer_depth_range[ridx].flatten(), 'm', where='post', linewidth=1.5, alpha=0.5, clip_on=False, label='Recipe')
        if ridx == 0:
            plt.legend(loc='lower right', fontsize=8)
    if fig_title:
        plt.suptitle(fig_title, fontweight='bold')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(os.path.join(save_path, f'{fig_title.replace(" ", "_").replace(".", "")}.png'), dpi=300)
    plt.show()

