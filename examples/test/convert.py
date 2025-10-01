import os
import sys

# import itk
import nrrd
import numpy as np

filename = sys.argv[1] if len(sys.argv) > 1 else './FinalVolume.npy'
dirname = os.path.dirname(filename)
outfile = os.path.join(dirname, 'FinalVolume.nrrd')

data = np.ascontiguousarray(np.load(filename))
print(data.shape)

threshold = 0.5
w = data > threshold
data[w] = 1
data[~w] = 0

nrrd.write(
    outfile,
    data,
    # header={
    #     'type': 'uint8',
    #     'dimension': 3,
    #     'sizes': data.shape,
    #     'encoding': 'gzip',
    # },
    index_order='C',
)

# image = itk.GetImageFromArray(
#     data,
#     # ttype=itk.Vector[itk.D, 3]
#     )
# itk.imwrite(image, os.path.join(dirname, 'FinalVolume.nrrd'))
