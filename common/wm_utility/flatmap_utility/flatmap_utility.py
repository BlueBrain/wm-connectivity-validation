import numpy
import pandas
from collections import OrderedDict
from scipy.spatial.transform import Rotation


def _flatmap_extent(fm, subsample=None):
    mx = fm.raw.max(axis=(0, 1, 2))
    if subsample is not None:
        mx = numpy.floor(mx / subsample).astype(int)
    return mx + 1


def _flat_coordinates_of_regions(names_region, fm, hier, ann, make_unique=True, subsample=None):
    reg_ids = set()
    for region in names_region:
        reg_ids.update(hier.find(region, "acronym", with_descendants=True))
    lst_ids = list(reg_ids)
    view3d = numpy.in1d(ann.raw.flat, lst_ids).reshape(ann.raw.shape)
    view2d = fm.raw[view3d]
    view2d = view2d[numpy.all(view2d >= 0, axis=1)]

    if subsample is not None:
        view2d = numpy.floor(view2d / subsample).astype(int)

    if not make_unique:
        return view2d
    mul = view2d[:, 0].max() + 1
    view_comb = view2d[:, 0] + mul * view2d[:, 1]
    unique_comb = numpy.unique(view_comb)
    return numpy.vstack([
        numpy.mod(unique_comb, mul),
        numpy.floor(unique_comb / mul)
    ]).transpose()


class Translation(object):
    def __init__(self, v):
        self._v = v

    def apply(self, other):
        return other + self._v

    def inv(self):
        return Translation(-self._v)


class Projection(object):
    def __init__(self, idx):
        self._idx = idx

    def apply(self, other):
        return other[:, self._idx]


class TwoDRotation(object):
    def __init__(self, M):
        self._M = M

    def apply(self, other):
        return numpy.dot(self._M, other.transpose()).transpose()

    def inv(self):
        return TwoDRotation(self._M.transpose())

    def expand(self):
        Mout = numpy.array([
            [self._M[0, 0], 0, self._M[0, 1]],
            [0, 1, 0],
            [self._M[1, 0], 0, self._M[1, 1]]
        ])
        return Rotation.from_matrix(Mout)


class Combination(object):
    def __init__(self, one, two):
        self._one = one
        self._two = two

    def apply(self, other):
        return self._two.apply(self._one.apply(other))

    def inv(self):
        return Combination(self._two.inv(), self._one.inv())


def flat_coordinate_frame(coordinates3d, fm, grouped=False):
    coords_flat = fm.lookup(coordinates3d)
    coord_frame = pandas.DataFrame(coordinates3d, index=pandas.MultiIndex.from_tuples(map(tuple, coords_flat),
                                                                                      names=["f_x", "f_y"]),
                                 columns=["x", "y", "z"])
    if grouped:
        return coord_frame.groupby(["f_x", "f_y"]).apply(lambda x: x.values)
    return coord_frame


def neuron_flat_coordinate_frame(circ, fm, grouped=False):
    coordinates3d = circ.cells.get(properties=["x", "y", "z"])
    coord_frame = flat_coordinate_frame(coordinates3d, fm)
    coord_frame["gid"] = coordinates3d.index.values
    if grouped:
        A = coord_frame[["x", "y", "z"]].groupby(["f_x", "f_y"]).apply(lambda x: x.values)
        B = coord_frame["gid"].groupby(["f_x", "f_y"]).apply(lambda x: x.values)
        return A, B
    return coord_frame


def voxel_flat_coordinate_frame(fm, in_voxel_indices=False, grouped=False):
    valid = numpy.all(fm.raw > -1, axis=-1)
    vxl_xyz = numpy.vstack(numpy.nonzero(valid)).transpose()
    vxl_flat = fm.raw[vxl_xyz[:, 0], vxl_xyz[:, 1], vxl_xyz[:, 2]]
    if not in_voxel_indices:
        vxl_xyz = vxl_xyz * fm.voxel_dimensions.reshape((1, -1)) + fm.offset.reshape((1, -1))
    vxl_frame = pandas.DataFrame(vxl_xyz, index=pandas.MultiIndex.from_tuples(map(tuple, vxl_flat),
                                                                              names=["f_x", "f_y"]),
                                 columns=["x", "y", "z"])
    if grouped:
        return vxl_frame.groupby(["f_x", "f_y"]).apply(lambda x: x.values)
    return vxl_frame


def flatmap_pixel_gradient(fm_or_frame):
    from ..wm_recipe_utility import colored_points_to_image
    if not isinstance(fm_or_frame, pandas.DataFrame):
        fm_or_frame = voxel_flat_coordinate_frame(fm_or_frame)
    per_pixel = fm_or_frame.groupby(["f_x", "f_y"])
    per_pixel_center = per_pixel.apply(lambda x: numpy.mean(x.values, axis=0))

    pxl_center_vol = colored_points_to_image(numpy.vstack(per_pixel_center.index.values),
                                             numpy.vstack(per_pixel_center.values))
    # Gradients thereof: When we go one step in the flat space, this is how many steps we going in global coordinates
    dx_dfx, dx_dfy = numpy.gradient(pxl_center_vol[:, :, 0])
    dy_dfx, dy_dfy = numpy.gradient(pxl_center_vol[:, :, 1])
    dz_dfx, dz_dfy = numpy.gradient(pxl_center_vol[:, :, 2])

    dfx = numpy.dstack([dx_dfx, dy_dfx, dz_dfx])
    dfy = numpy.dstack([dx_dfy, dy_dfy, dz_dfy])
    return dfx, dfy


def _find_rotation_(v_x, v_y):
    if numpy.any(numpy.isnan(v_x)):
        if numpy.any(numpy.isnan(v_y)):
            return TwoDRotation(numpy.identity(2)), 2.0
        vv = numpy.hstack([v_y, [[0]]])
        vtgt = numpy.array([[0, 1, 0]])
    elif numpy.any(numpy.isnan(v_y)):
        vv = numpy.hstack([v_x, [[0]]])
        vtgt = numpy.array([[1, 0, 0]])
    else:
        vv = numpy.hstack([numpy.vstack([v_x, v_y]), [[0], [0]]])
        vtgt = numpy.array([[1, 0, 0], [0, 1, 0]])
    vv = vv / numpy.linalg.norm(vv, axis=1, keepdims=True)
    res = Rotation.align_vectors(vtgt, vv)
    M = res[0].as_matrix()
    return TwoDRotation(M[:2, :2]), res[1]


def per_pixel_coordinate_transformation(fm, orient, from_system="global", to_system="rotated"):
    """
    Systems:
    global: The global coordinate system in um or voxel indices.
    localized: The global coordinate system, but origin moved to the pixel center
    rotated: A local coordinate system, origin at the pixel center, y-axis rotated vertical
    rotated_flat: A local coordinate system, origin at the pixel center, y-axis rotated vertical and then flattened away
    through parallel projection
    subpixel: A local coordinate system, origin at the pixel center, y-axis rotated vertical and then flattened away
    through parallel projection, x and z axes oriented like the flat-x and flat-y axes of the flat map.
    subpixel_depth: A local coordinate system, origin at the pixel center, y-axis rotated vertical,
    x and z axes oriented like the flat-x and flat-y axes of the flat map.
    """
    lst_systems = ["global", "localized", "rotated", "rotated_flat", "subpixel", "subpixel_depth"]
    try:
        tgt_tf = (lst_systems.index(from_system), lst_systems.index(to_system))
    except ValueError:
        raise ValueError("from_system and to_system must be in: {0}, but you provided {1}".format(lst_systems,
                                                                                                  (from_system,
                                                                                                   to_system)))
    invalid_combos = [(2, 3), (3, 2), (3, 1), (3, 0), (4, 2), (4, 1), (4, 0), (3, 5), (4, 5), (5, 3), (5, 4)]
    if tgt_tf in invalid_combos:
        raise ValueError("Invalid combination!")
    if tgt_tf[0] == tgt_tf[1]:
        raise ValueError("Identity transformation not supported")

    vxl_frame = voxel_flat_coordinate_frame(fm)
    per_pixel = vxl_frame.groupby(["f_x", "f_y"])

    per_pixel_negative_center = per_pixel.apply(lambda x: -numpy.mean(x.values, axis=0))
    global2localized = per_pixel_negative_center.apply(Translation)
    if tgt_tf == (0, 1): return global2localized
    if tgt_tf == (1, 0): return global2localized.apply(lambda x: x.inv())

    per_pixel_orient = per_pixel_negative_center.apply(lambda x: orient.lookup(-x))
    localized2rotated = per_pixel_orient.apply(lambda o_vec: Rotation.from_quat(numpy.hstack([o_vec[1:],
                                                                                       o_vec[0:1]])).inv())
    if tgt_tf == (1, 2): return localized2rotated
    if tgt_tf == (2, 1): return localized2rotated.apply(lambda x: x.inv())

    global2rotated = global2localized.combine(localized2rotated, Combination)
    if tgt_tf == (0, 2): return global2rotated
    if tgt_tf == (2, 0): return global2rotated.apply(lambda x: x.inv())

    tf_to_local_flat = Projection([0, 2])
    global2rotflat = global2rotated.apply(lambda base_tf: Combination(base_tf, tf_to_local_flat))
    if tgt_tf == (0, 3): return global2rotflat
    if tgt_tf == (3, 0): return global2rotflat.apply(lambda x: x.inv())

    localized2rotflat = localized2rotated.apply(lambda base_tf: Combination(base_tf, tf_to_local_flat))
    if tgt_tf == (1, 3): return localized2rotflat

    dfx, dfy = flatmap_pixel_gradient(vxl_frame)
    dfx_frame = per_pixel_negative_center.index.to_frame().apply(lambda x: dfx[x["f_x"], x["f_y"]].reshape((1, -1)),
                                                            axis=1)
    dfy_frame = per_pixel_negative_center.index.to_frame().apply(lambda x: dfy[x["f_x"], x["f_y"]].reshape((1, -1)),
                                                            axis=1)
    # Above gradient vectors are in "localized" space. Convert to rotated_flat
    dfx_frame = localized2rotflat.combine(dfx_frame, lambda a, b: a.apply(b))
    dfy_frame = localized2rotflat.combine(dfy_frame, lambda a, b: a.apply(b))

    # We now know the directions of neighboring pixel in the rotated_flat systems
    # Figure out a rotation that transforms the direction vectors to the x or y-axes
    rotflat2pixel_err = dfx_frame.combine(dfy_frame, _find_rotation_)
    rotflat2pixel = rotflat2pixel_err.apply(lambda x: x[0])
    err = rotflat2pixel_err.apply(lambda x: x[1])
    print("Rotation errors: min: {0}, median: {1}, mean: {2}, std: {3}, max: {4}".format(
        err.min(), err.median(), err.mean(), err.std(), err.max()
    ))
    if tgt_tf == (3, 4): return rotflat2pixel
    if tgt_tf == (4, 3): return rotflat2pixel.apply(lambda x: x.inv())
    if tgt_tf == (2, 4): return rotflat2pixel.apply(lambda base_tf: Combination(tf_to_local_flat, base_tf))
    if tgt_tf == (1, 4): return localized2rotflat.combine(rotflat2pixel, Combination)
    if tgt_tf == (0, 4): return global2rotflat.combine(rotflat2pixel, Combination)

    rot2pixel = rotflat2pixel.apply(lambda x: x.expand())
    if tgt_tf == (2, 5): return rot2pixel
    if tgt_tf == (5, 2): return rot2pixel.apply(lambda x: x.inv())
    if tgt_tf == (1, 5): return localized2rotated.combine(rot2pixel, Combination)
    if tgt_tf == (5, 1): return rot2pixel.combine(localized2rotated, lambda a, b: Combination(a.inv(), b.inv()))
    if tgt_tf == (0, 5): return global2rotated.combine(rot2pixel, Combination)
    if tgt_tf == (5, 0): return rot2pixel.combine(global2rotated, lambda a, b: Combination(a.inv(), b.inv()))
    raise ValueError("This should never happen!")


def supersample_flatmap(fm, orient, pixel_sz=34.0):
    import voxcell
    vxl_frame = voxel_flat_coordinate_frame(fm, grouped=True)
    tf = per_pixel_coordinate_transformation(fm, orient, to_system="subpixel")
    subpixel_loc = vxl_frame.combine(tf, lambda a, b: b.apply(a))
    final_loc = subpixel_loc.index.to_series().combine(subpixel_loc,
                                                       lambda a, b: pixel_sz * numpy.array(a) + b)
    final_loc_arr = numpy.vstack(final_loc.values)
    vxl_loc = numpy.vstack(vxl_frame.values)
    out_raw = -numpy.ones_like(fm.raw, dtype=float)
    out_raw[vxl_loc[:, 0], vxl_loc[:, 1], vxl_loc[:, 2]] = final_loc_arr
    return voxcell.VoxelData(out_raw, fm.voxel_dimensions, offset=fm.offset)


def supersampled_neuron_locations(circ, fm, orient, pixel_sz=34.0):
    nrn_loc_frame, nrn_gid_frame = neuron_flat_coordinate_frame(circ, fm, grouped=True)
    tf = per_pixel_coordinate_transformation(fm, orient, to_system="subpixel")
    idxx = nrn_loc_frame.index.intersection(tf.index)

    res = tf[idxx].combine(nrn_loc_frame[idxx], lambda a, b: a.apply(b))
    final = res.index.to_series().combine(res, lambda a, b: numpy.array(a) * pixel_sz + b)
    final_frame = numpy.vstack(final.values)
    out = pandas.DataFrame(final_frame,
                           columns=["flat x", "flat y"],
                           index=numpy.hstack(nrn_gid_frame[idxx].values))
    return out


def flat_coordinates_of_regions(names_regions, fm, *args, make_unique=False, subsample=None):
    if len(args) == 2:
        return _flat_coordinates_of_regions(names_regions, fm, *args, make_unique=make_unique, subsample=subsample)
    elif len(args) == 1:
        circ = args[0]
        atlas = circ.atlas
        hier = atlas.load_region_map()
        ann = atlas.load_data("brain_regions")
        return _flat_coordinates_of_regions(names_regions, fm, hier, ann, make_unique=make_unique, subsample=subsample)
    else:
        raise ValueError()


def flat_region_image(lst_regions, fm, *args, extent=None, subsample=None):
    if extent is None:
        extent = _flatmap_extent(fm, subsample=subsample)
    xbins = numpy.arange(extent[0] + 1)
    ybins = numpy.arange(extent[1] + 1)
    counters = []
    for reg in lst_regions:
        if not isinstance(reg, list):
            reg = [reg]
        A = flat_coordinates_of_regions(reg, fm, *args, make_unique=False, subsample=subsample)
        H = numpy.histogram2d(A[:, 0], A[:, 1], bins=(xbins, ybins))[0]
        counters.append(H)

    region_lookup = numpy.argmax(numpy.dstack(counters), axis=2)
    region_lookup[numpy.sum(numpy.dstack(counters), axis=2) == 0] = -1
    return region_lookup
