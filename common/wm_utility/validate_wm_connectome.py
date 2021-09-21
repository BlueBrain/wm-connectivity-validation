import pandas
import numpy
import os


def read_pixel_to_pixel_connectivity(fn):
    data = pandas.read_pickle(fn)
    src_coord_frame = data["Source region"].apply(lambda x: (int(x.split(",")[0]), int(x.split(",")[1])))
    src_coord_frame = pandas.DataFrame.from_records(src_coord_frame, index=src_coord_frame.index,
                                                    columns=["src x", "src y"])

    tgt_coord_frame = data["Target region"].apply(lambda x: (int(x.split(",")[0]), int(x.split(",")[1])))
    tgt_coord_frame = pandas.DataFrame.from_records(tgt_coord_frame, index=tgt_coord_frame.index,
                                                    columns=["tgt x", "tgt y"])
    full_index = pandas.concat([src_coord_frame, tgt_coord_frame], axis=1)
    data.index = pandas.MultiIndex.from_frame(full_index)

    density = data.loc[(full_index != -1).all(axis=1).values]["Density"]
    return density


def pixel_to_pixel2array(density_series):
    full_index = density_series.index.to_frame()
    n_x = full_index.max(axis=0)["src x"] + 1
    n_y = full_index.max(axis=0)["src y"] + 1
    img4d = numpy.zeros((n_x, n_y, n_x, n_y), dtype=float)
    for idx, v in density_series.items():
        img4d[idx] = v
    return img4d


def anterograde_connectivity_rgb_frame(fn_pixel_to_pixel, rgb_img):
    density_series = read_pixel_to_pixel_connectivity(fn_pixel_to_pixel)
    src_idx = density_series.index.to_frame()[["src x", "src y"]].apply(lambda x: tuple(x.values), axis=0)
    src_cols = pandas.DataFrame(rgb_img[src_idx["src x"], src_idx["src y"]], index=density_series.index,
                                columns=["src r", "src g", "src b"])

    def func(lst_in):
        arr_in = numpy.vstack(lst_in.values)
        out = numpy.nansum(arr_in[:, 0:1] * arr_in[:, 1:], axis=0)
        out = out / numpy.nansum(arr_in[:, 0])
        return out

    tgt_cols = pandas.concat([density_series, src_cols], axis=1).groupby(["tgt x", "tgt y"]).apply(func)
    return tgt_cols


def anterograde_connectivity_rgb_image(fn_pixel_to_pixel, rgb_img_in):
    from wm_utility.wm_recipe_utility import colored_points_to_image
    frame = anterograde_connectivity_rgb_frame(fn_pixel_to_pixel, rgb_img_in)
    return colored_points_to_image(frame.index.to_frame().values, frame.values)
