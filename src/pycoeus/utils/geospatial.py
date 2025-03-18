"""Geosptial related utilities."""

from typing import Literal
import geopandas as gpd
import numpy as np
import xarray as xr


def get_label_array(
    input_data: xr.DataArray,
    geom_pos: gpd.GeoSeries,
    geom_neg: gpd.GeoSeries,
    compute_mode: Literal["normal", "parallel", "safe"] = "normal",
) -> xr.DataArray:
    """Generate a label array for binary classification from pos/neg geometries.

    The function overlays pos/neg geometries on a template raster,
    then fills the overlapped area with the label value,
    leaving the rest of the raster with -1.

    There are three values filled:
    - 1: positive label
    - 0: negative label
    - -1: unclassified

    One needs to call geom_to_label_array at least twice to get a
    complete label array for a binary classification task.

    There are two modes of execution:
    - normal: Assumes the label array fits in memory
    - parallel or safe: Assumes the label array is too large to fit in memory, do block processing.
        The difference between parallel and safe is safe only uses single thread, which will be
        configured in dask schedular.

    :param xr.DataArray input_data: input features as a template raster
    :param gpd.GeoSeries geom: Geometry of the label
    :param Literal["normal", "parallel", "safe"] mode: Mode of execution, defaults to "normal"
    :return xr.DataArray: Generated label array
    """
    # If the template raster has a "band" dimension
    # Select the first band, since label will be 2D
    if "band" in input_data.dims:
        input_template = input_data.isel(band=0).drop_vars("band")


    positive_labels = _geom_to_label_array(input_template, geom_pos, 1, compute_mode=compute_mode)
    negative_labels = _geom_to_label_array(input_template, geom_neg, 0, compute_mode=compute_mode)

    labels = -(positive_labels * negative_labels)  # Combine positive and negative labels

    # repeat to match the input_template
    return labels.expand_dims(dim={"band": input_data.sizes["band"]})


def _geom_to_label_array(
    input_template: xr.DataArray,
    geom: gpd.GeoSeries,
    value: Literal[0, 1],
    compute_mode: Literal["normal", "parallel", "safe"] = "normal",
) -> xr.DataArray:
    """Generate a label array from a geometry."""
    # Make a template from the shape of the template raster
    # Fill it with the label value
    labels_template = xr.full_like(input_template, fill_value=value, dtype=np.int32)

    # Set the nodata value to -1 indicating other classes
    labels_template = labels_template.rio.write_nodata(-1)

    match compute_mode:
        # Assumning labels_template can fit in memory
        case "normal":
            label_array = labels_template.rio.clip(geom.geometry, drop=False)
        # Block processing for large rasters
        case "parallel" | "safe":
            label_array = xr.map_blocks(
                lambda raster, geom: raster.rio.clip(geom, drop=False),
                labels_template,
                args=(geom.geometry,),
                template=labels_template,
            )
        case _:
            msg = f"Mode {compute_mode} not supported"
            raise ValueError(msg)

    return label_array
