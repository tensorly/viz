# -*- coding: utf-8 -*-

__author__ = "Marie Roald & Yngve Mardal Moe"

import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr
from scipy.io import loadmat

from .utils import cp_to_tensor

__all__ = ["load_aminoacids", "load_oslo_city_bike", "download_city_bike"]

DATASET_PARENT = Path(__file__).parent / "datasets"
DOWNLOADED_PARENT = DATASET_PARENT / "downloads"


class RemoteZip:
    def __init__(self, url):
        req = requests.get(url)
        self.zip = zipfile.ZipFile(BytesIO(req.content))

    @property
    def contents(self):
        return [f.filename for f in self.zip.infolist()]

    def extract_file(self, filename):
        file_contents = self.zip.read(filename)
        data = BytesIO(file_contents)
        return data

    def extract_all(self):
        return {filename: self.extract_file(filename) for filename in self.contents}


def load_aminoacids(save_data=True):
    """Load the Aminoacids dataset from "PARAFAC. Tutorial and applications" by Rasmus Bro :cite:p:`bro1997parafac`.

    This is a fluoresence spectroscopy dataset well suited for PARAFAC analysis.
    The data stems from five samples with different amounts of the aminoacids tyrosine, tryptophan and phenylalanine.

    See here for more information about the data:
        http://models.life.ku.dk/Amino_Acid_fluo
    Or archived version here:
        https://web.archive.org/web/20210413050155/http://models.life.ku.dk/Amino_Acid_fluo

    Parameters
    ----------
    save_data : bool (default=True)
        If ``True``, then the dataset is saved in ``tlviz.data.DOWNLOAD_PARENT`` to avoid having to download
        it in the future.

    Returns
    -------
    xarray.DataArray
        The Aminoacids dataset
    """
    print("Loading Aminoacids dataset from:")
    print(
        "Bro, R, PARAFAC: Tutorial and applications, Chemometrics and Intelligent Laboratory Systems,"
        + " 1997, 38, 149-171"
    )

    filename = DOWNLOADED_PARENT / "aminoacids.nc4"
    if filename.is_file():
        return xr.open_dataarray(filename)

    aminoacids_zip = RemoteZip("http://models.life.ku.dk/sites/default/files/Amino_Acid_fluo.zip")
    matlab_variables = loadmat(aminoacids_zip.extract_file("amino.mat"))

    I, K, J = matlab_variables["DimX"].squeeze().astype(int)
    X = matlab_variables["X"].reshape(I, J, K)
    emission_frequencies = matlab_variables["EmAx"].squeeze()
    excitation_frequencies = matlab_variables["ExAx"].squeeze()
    coords_dict = {
        "Sample": list(range(I)),
        "Emission frequency": emission_frequencies,
        "Excitation frequency": excitation_frequencies,
    }
    dims = ["Sample", "Excitation frequency", "Emission frequency"]

    data = xr.DataArray(X, dims=dims, coords=coords_dict)

    if save_data:
        DOWNLOADED_PARENT.mkdir(exist_ok=True, parents=True)
        data.to_netcdf(filename)

    return data


def load_oslo_city_bike():
    """Download bike data from the bike sharing system in Oslo 2020-2021.

    The dataset is a five-way tensor with modes:

     * Bike station id
     * Year
     * Month
     * Day of week
     * Hour of day

    where the elements corresponds to the number of trips ending in a given
    time period and station. For example, the index ``377, 2020, 5, 1, 13`` corresponds
    to the number of trips that ended at station 377 at 1pm (CET time) on Mondays in May of 2020.

    The data was collected using the open API of https://oslobysykkel.no/en/open-data.

    Returns
    -------
    xarray.DataArray
        Data array whose data is the dataset and coordinates are bike station ID, year, month, day of week
        and hour of day. There are also three metadata-coordinates along the "Bike station ID"-axis containing
        latitudes, longitudes and station names.
    """
    nc4_file = DATASET_PARENT / "oslo_bike.nc4"
    return xr.load_dataarray(nc4_file)


def download_city_bike(source="oslobysykkel.no", years=(2020, 2021)):
    """Download bike data from the bike sharing system in Oslo.

    The dataset is a five-way tensor with modes:

     * Bike station id
     * Year
     * Month
     * Day of week
     * Hour of day

    where the elements corresponds to the number of trips ending in a given
    time period and station. For example, the index ``377, 2020, 5, 1, 13`` corresponds
    to the number of trips that ended at station 377 at 1pm (CET time) on Mondays in May of 2020.

    The data was collected using the open urbansharing API.

    Parameters
    ----------
    source : str
        String to download data from. The API endpoint becomes:
        https://data.urbansharing.com/{source}/trips/v1/{year}/{month:02d}.csv
        Some end-points that are known to work are:

         * oslobysykkel.no
         * bergenbysykkel.no
         * trondheimbysykkel.no

    years : iterable of int
        The years to download data from.

    Returns
    -------
    xarray.DataArray
        Data array whose data is the dataset and coordinates are bike station ID, year, month, day of week
        and hour of day. There are also three metadata-coordinates along the "Bike station ID"-axis containing
        latitudes, longitudes and station names.
    """
    end_data = []

    lat = {}
    lon = {}
    name = {}
    for year in years:
        print(f"Loading {year}", flush=True)
        for month in range(12):
            month += 1
            df = pd.read_csv(f"https://data.urbansharing.com/{source}/trips/v1/{year}/{month:02d}.csv")
            # station_names = sorted(set(df["start_station_name"]) | set(df["end_station_name"]))

            df["ended_at"] = df["ended_at"].map(lambda x: x if "." in x else ".000000+".join(x.split("+")))
            df["ended_at"] = pd.to_datetime(df["ended_at"], format="%Y-%m-%d %H:%M:%S.%f+00:00")
            df["ended_at"] = df["ended_at"].dt.tz_localize("UTC")
            df["ended_at"] = df["ended_at"].dt.tz_convert("CET")

            end_time = df["ended_at"].dt

            df["trip"] = 1
            df["Day"] = end_time.day
            df["Day of week"] = end_time.dayofweek
            df["Hour"] = end_time.hour
            df["Month"] = end_time.month
            df["Year"] = end_time.year
            df = df.rename({"end_station_id": "End station ID"}, axis=1)

            for _, row in df.iterrows():
                lat[row["End station ID"]] = row["end_station_latitude"]
                lon[row["End station ID"]] = row["end_station_longitude"]
                name[row["End station ID"]] = row["end_station_name"]

            end = df.groupby(["End station ID", "Year", "Month", "Day of week", "Hour"]).sum()[["trip"]]
            end_data.append(end)

    grouped = pd.concat(end_data).groupby(level=(0, 1, 2, 3, 4)).sum()

    dataset = xr.Dataset.from_dataframe(grouped).to_array().squeeze()

    # Drop trips that started in the last year of interest and ended the year after
    unwanted_years = set(dataset.coords["Year"].values) - set(years)
    for year in unwanted_years:
        dataset = dataset.drop(year, "Year")

    # Set equal to 0 for all hours with no trips
    dataset.values[np.isnan(dataset.values)] = 0

    # Add station metadata
    lat = [lat[station_id.item()] for station_id in dataset.coords["End station ID"]]
    lon = [lon[station_id.item()] for station_id in dataset.coords["End station ID"]]
    name = [name[station_id.item()] for station_id in dataset.coords["End station ID"]]
    dataset = xr.DataArray(
        dataset.data,
        coords={
            **{key: value for key, value in dataset.coords.items() if key != "variable"},
            "lat": (("End station ID",), lat),
            "lon": (("End station ID",), lon),
            "name": (("End station ID",), name),
        },
        dims=dataset.dims,
        name="Bike trips",
    )
    return dataset


def simulated_random_cp_tensor(shape, rank, noise_level=0.1, labelled=False, seed=None):
    """Create a random noisy CP tensor.

    The factor matrices and weights have uniformly distributed elements and
    the noise is normally distributed with magnitude ``noise_level * tl.norm(X)``,
    where ``X`` is the datatensor represented by the decomposition.

    Parameters
    ----------
    shape : iterable of ints
        The shape of the data tensor
    rank : int
        The number of components
    noise_level : float
        Relative magnitude of the noise compared to the magnitude of the data.
    seed : {None, int, array_like[ints], np.random.SeedSequence, np.random.BitGenerator, np.random.Generator}
        Seed for numpy random number generator

    Returns
    -------
    cp_tensor
        CP tensor that represents the simulated dataset

    np.ndarray
        Dense tensor with noise added
    """
    rng = np.random.default_rng(seed)
    weights = rng.random(size=rank)
    factors = [rng.random(size=(length, rank)) for length in shape]
    if labelled:
        for i, factor in enumerate(factors):
            factors[i] = pd.DataFrame(factor)
            factors[i].index.name = f"Mode {i}"
    cp = weights, factors

    X = cp_to_tensor(cp)
    noise = rng.standard_normal(size=shape)
    X_noisy = X + np.linalg.norm(X) * noise_level * noise / np.linalg.norm(noise)

    return cp, X_noisy


# TODO: Add more example datasets
# TODO: Enron data
# TOTEST: data.py
