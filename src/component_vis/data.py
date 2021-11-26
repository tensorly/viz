import zipfile
from io import BytesIO
import requests
from scipy.io import loadmat
import xarray as xr


__all__ = ["load_aminoacids"]


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


def load_aminoacids():
    # TODO: Docstring for load_aminoacids
    print(
        'Loading Aminoacids dataset from "PARAFAC. Tutorial and applications" by Rasmus Bro'
    )
    aminoacids_zip = RemoteZip(
        "http://models.life.ku.dk/sites/default/files/Amino_Acid_fluo.zip"
    )
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

    return xr.DataArray(X, dims=dims, coords=coords_dict)


# TODO: Add more example datasets
