import numpy
from common.wm_utility.flatmap_utility import flat_coordinates_of_regions as two_d_view_of
from common.wm_utility.flatmap_utility import flat_region_image
from common.wm_utility.flatmap_utility.flatmap_utility import _flatmap_extent


def _loader(recipe):
    if isinstance(recipe, str):
        import yaml
        with open(recipe, "r") as fid:
            mp = yaml.load(fid, Loader=yaml.SafeLoader)
        return mp
    return recipe


def lookup_population_in_recipe(nm, recipe):
    tmp = [x for x in recipe["populations"] if x["name"] == nm]
    assert len(tmp) == 1
    regions = tmp[0]['atlas_region']
    if isinstance(regions, list):
        return [region["name"] for region in regions]
    return [regions['name']]


def twod2rgb(flat_coords, x, y):
    pts = flat_coords - numpy.array([[x[2], y[2]]])
    T = numpy.array([[x[0] - x[2], x[1] - x[2]],
                    [y[0] - y[2], y[1] - y[2]]])
    Tinv = numpy.linalg.inv(T)
    l_0_1 = numpy.dot(Tinv, pts.transpose())
    l = numpy.vstack([l_0_1, 1.0 - numpy.sum(l_0_1, axis=0, keepdims=True)]).transpose()
    l[l < 0] = 0.0; l[l > 1] = 1.0
    return l


def colored_points_to_image(flat_coords, cols, extent=None):
    flat_coords = flat_coords.astype(int)
    if extent is None:
        extent = flat_coords.max(axis=0) + 1
    img = numpy.NaN * numpy.ones(tuple(extent) + (3, ))
    for xy, col in zip(flat_coords, cols):
        img[xy[0], xy[1], :] = col
    return img


def recipe2source_rgb(recipe, fm, *args, img_extent=None, return_lookup=True, subsample=None):
    recipe = _loader(recipe)

    if img_extent is None:
        img_extent = _flatmap_extent(fm, subsample=subsample)

    region_img_dict = {}
    for proj in recipe["projections"]:
        src_str = proj["source"].split("_")[0]
        if src_str in region_img_dict:
            continue
        pop_strs = lookup_population_in_recipe(proj["source"], recipe)

        pts_src = two_d_view_of(pop_strs, fm, *args, make_unique=True, subsample=subsample)
        x = numpy.array(proj["mapping_coordinate_system"]["x"])
        y = numpy.array(proj["mapping_coordinate_system"]["y"])
        if subsample is not None:
            x = x / subsample; y = y / subsample
        col_src = twod2rgb(pts_src, x, y)
        region_img_dict[src_str] = (
            pop_strs,
            colored_points_to_image(pts_src, col_src, extent=img_extent)
        )
    kk = region_img_dict.keys()
    lst_all_regions = [region_img_dict[_k][0] for _k in kk]
    region_lookup = flat_region_image(lst_all_regions, fm, *args, extent=img_extent, subsample=subsample)
    img_out = numpy.zeros(tuple(img_extent) + (3,), dtype=float)

    for i, k in enumerate(kk):
        valid = region_lookup == i
        img_out[valid, :] = region_img_dict[k][1][valid, :]
    if return_lookup:
        return img_out, (region_lookup, lst_all_regions)
    return img_out
