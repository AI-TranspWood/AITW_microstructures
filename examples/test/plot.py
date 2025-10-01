import sys

import matplotlib.pyplot as plt
import numpy as np
from mayavi import mlab
from tvtk.util import ctf

file = sys.argv[1] if len(sys.argv) > 1 else './FinalVolume.npy'

data = np.load(file)
print(data.shape)

threshold = 0.5
w = data > threshold
data[w] = 1
data[~w] = 0


# # Plot using mayavi
# mayavi.mlab.contour3d(data, contours=10, opacity=0.5)
# mayavi.mlab.axes()
# mayavi.mlab.show()

# Plot volume slicer using mayavi where above threshold is 1 and below is 0
mlab.figure(bgcolor=(1.0, 1.0, 1.0), size=(1600, 1600))
src = mlab.pipeline.scalar_field(data)
src.update_image_data = True
# mlab.pipeline.image_plane_widget(
#     src,
#     plane_orientation='z_axes',
#     slice_index=data.shape[0] // 2,
# )
volume = mlab.pipeline.volume(src, vmin=0, vmax=1)

c = ctf.save_ctfs(volume._volume_property)
c['rgb'] = plt.get_cmap('gray')(np.arange(2))
ctf.load_ctfs(c, volume._volume_property)

volume.update_ctf = True


mlab.axes()
mlab.show()
