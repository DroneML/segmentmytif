"""Geosptial related utilities."""

from typing import Literal
import geopandas as gpd
import numpy as np
import xarray as xr


def geom_to_label_array(
    ras_template: xr.DataArray,
    geom: gpd.GeoSeries,
    value: Literal[0, 1],
    mode: Literal["normal", "parallel", "safe"] = "normal",
) -> xr.DataArray:
    """Generate a label array for binary classification from a geometry.

    The function overlays a geometry on a template raster,
    then fills the overlapped area with the label value,
    leaving the rest of the raster with -1.

    The label array should have three classes:
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

    :param xr.DataArray ras_template: template raster with disired shape of the label array
    :param gpd.GeoSeries geom: Geometry of the label
    :param Literal[0, 1] value: Label values
    :param Literal["normal", "parallel", "safe"] mode: Mode of execution, defaults to "normal"
    :return xr.DataArray: Generated label array
    """
    # If the template raster has a "band" dimension
    # Select the first band, since label will be 2D
    if "band" in ras_template.dims:
        ras_template = ras_template.isel(band=0)

    # Make a template from the shape of the template raster
    # Fill it with the label value
    labels_template = xr.full_like(ras_template, fill_value=value, dtype=np.int32)

    # Set the nodata value to -1 indicating other classes
    labels_template = labels_template.rio.write_nodata(-1)

    match mode:
        # Assumning labels_template can fit in memory
        case "normal":
            label_array = labels_template.rio.clip(geom, drop=False)
        # Block processing for large rasters
        case "parallel" | "safe":
            label_array = xr.map_blocks(
                lambda raster, geom: raster.rio.clip(geom, drop=False),
                labels_template,
                args=(geom.geometry,),
                template=labels_template,
            )
        case _:
            msg = f"Mode {mode} not supported"
            raise ValueError(msg)

    return label_array
