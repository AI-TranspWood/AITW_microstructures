import os
import sys

import nrrd
import numpy as np
import numpy.typing as npt


def input_npy(fpath: str) -> npt.NDArray:
    """Load a numpy file."""
    return np.load(fpath)

def input_nrrd(fpath: str) -> npt.NDArray:
    """Load a nrrd file."""
    return nrrd.read(fpath, index_order='C')[0]

def input_vti(fpath: str) -> npt.NDArray:
    """Load a vti file."""
    try:
        import vtk
    except ImportError:
        sys.exit('ERROR: Install VTK to use VTI files')
    from vtk.util import numpy_support
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(fpath)
    reader.Update()
    vd = reader.GetOutput()

    dims = reader.GetOutput().GetDimensions()

    scalars = vd.GetPointData().GetScalars() or vd.GetPointData().GetArray(0)

    return numpy_support.vtk_to_numpy(scalars).reshape(dims, order='F').astype(np.float32)

def output_npy(fpath: str, data: npt.NDArray):
    """Save a numpy file."""
    np.save(fpath, data)

def output_nrrd(fpath: str, data: npt.NDArray):
    """Save a nrrd file."""
    nrrd.write(fpath, data, index_order='C')

def output_vti(fpath: str, data: npt.NDArray):
    """Save a vti file."""
    try:
        import vtk
    except ImportError:
        sys.exit('ERROR: Install VTK to use VTI files')
    from vtk.util import numpy_support

    data = data.astype(np.uint8)  # Ensure data is uint8 for VTI output
    dims = data.shape
    data_flat = data.flatten(order='F')

    image_data = vtk.vtkImageData()
    image_data.SetDimensions(dims)
    image_data.SetSpacing(1.0, 1.0, 1.0)
    image_data.SetOrigin(0.0, 0.0, 0.0)

    vtk_array = numpy_support.numpy_to_vtk(data_flat, deep=True, array_type=vtk.VTK_FLOAT)
    vtk_array.SetName('ImageFile')
    image_data.GetPointData().SetScalars(vtk_array)

    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(fpath)
    writer.SetInputData(image_data)
    writer.Write()

input_funcs = {
    'npy': input_npy,
    'nrrd': input_nrrd,
    'vti': input_vti,
}

output_funcs = {
    'npy': output_npy,
    'nrrd': output_nrrd,
    'vti': output_vti,
}

def read_volume(fpath: str) -> npt.NDArray:
    """Read a volume from a file."""
    ext = os.path.splitext(fpath)[1][1:]  # Get the file extension without the dot
    if ext not in input_funcs:
        raise ValueError(f"Unsupported file format: {ext}. Supported formats: {list(input_funcs.keys())}")
    return input_funcs[ext](fpath)

def write_volume(fpath: str, data: npt.NDArray):
    """Write a volume to a file."""
    ext = os.path.splitext(fpath)[1][1:]  # Get the file extension without the dot
    if ext not in output_funcs:
        raise ValueError(f"Unsupported file format: {ext}. Supported formats: {list(output_funcs.keys())}")
    output_funcs[ext](fpath, data)

__all__ = [
    'input_npy',
    'input_nrrd',
    'input_vti',
    'output_npy',
    'output_nrrd',
    'output_vti',
    'read_volume',
    'write_volume',
]
