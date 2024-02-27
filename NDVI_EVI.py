#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#creating an animation from list of various indices
# import glob
# from pathlib import Path
# image_path=Path(path_bandcomposites)
# images=list(image_path.glob('*.png'))
# image_list=[]
# for file_name in images:
#     image_list.append(imageio.inread(file_name))

# gif = imageio.mimwrite('/directory/'+'image.gif',image_list, fps=2)


# ![dfe6ff10-19ca-11ee-8d5b-8f954b8ec342.gif](attachment:f38814bf-542c-438d-ba15-3e0710d85256.gif)

# In[ ]:





# In[7]:


import rasterio 
import numpy as np
import matplotlib.pyplot as plt

# Open the Landsat image with 7 bands
with rasterio.open("E:/errors in qgis/LANDSAT2/2003.tif") as src:
    bands = src.read()

    # Calculate the total sum of all bands
    total = np.zeros(bands[0].shape, dtype=np.float64)
    for band in bands:
        total += band

    # Divide by the number of bands to get the average
    total /= len(bands)

    # Save the profile information before exiting the `with` block
    profile = src.profile

# Write the averaged band as a raster file
profile.update(dtype=rasterio.float64, count=1, compress='lzw')

# Specify the output file name
output_filename = "E:/errors in qgis/LANDSAT2/2003_composite.tif"

with rasterio.open(output_filename, 'w', **profile) as dst:
    dst.write(total.astype(rasterio.float64), 1)

# Plot the output image
plt.imshow(total, cmap='viridis')
plt.colorbar(label='Pixel Value')
plt.title('Composite Image')
plt.show()


# In[10]:


import rasterio 
from rasterio.plot import show

# Open the Landsat image with 7 bands
with rasterio.open("E:/errors in qgis/LANDSAT2/2003.tif") as src:
    show(src)


# # NDVI CALCULATIONS

# In[11]:


import rasterio 

# Open the Landsat image with 7 bands
with rasterio.open("E:/errors in qgis/LANDSAT2/2003.tif") as src:
    # Get the band indexes
    band_indexes = src.indexes

# Print the band indexes as a list
print("Band Indexes:", list(band_indexes))


# In[40]:


import rasterio 
import numpy as np
import matplotlib.pyplot as plt

# Open the Landsat image with 7 bands
with rasterio.open("E:/errors in qgis/LANDSAT2/2003.tif") as src:
    # Read the near-infraredt (NIR) and red bands
    nir = src.read(5,masked=True)
    red = src.read(4,masked=True)

    # Calculate NDVI
    ndvi = (nir - red) / (nir + red)

    # Plot NDVI
    plt.figure(figsize=(10, 10))
    plt.imshow(ndvi, cmap=color_ramp, vmin=-1, vmax=1)
    plt.colorbar(label='NDVI')
    plt.title('Normalized Difference Vegetation Index (NDVI)')
    plt.show

output_filend = "E:/errors in qgis/LANDSAT2/forgif/2003NDVI.tif"

    # Write the EVI to a raster file with the same geospatial metadata as the input Landsat image
with rasterio.open(
    output_filend,
    'w',
    driver='GTiff',
     height=src.height,
    width=src.width,
    count=1,
    dtype=rasterio.float32,
    crs=src.crs,
    transform=src.transform,
    nodata=None
) as dst:
    dst.write(evi, 1)
    print("NDVI calculation completed and saved to:", output_filend)



# In[50]:


import rasterio 
import numpy as np
import matplotlib.pyplot as plt

# Open the Landsat image with 7 bands
with rasterio.open("E:/errors in qgis/LANDSAT2/2003.tif") as src:
    # Read the near-infrared (NIR) and red bands
    nir = src.read(4, masked=True)
    red = src.read(3, masked=True)

    # Calculate NDVI
    ndvi = (nir - red) / (nir + red)

# Define the color ramp for NDVI visualization
# Green to represent healthy vegetation, gray for non-vegetated areas, and brown for negative NDVI values (water, clouds, etc.)
color_ramp = plt.cm.RdYlGn  # Red-Yellow-Green colormap

# Plot the NDVI results
plt.figure(figsize=(10, 6))
plt.imshow(ndvi, cmap=color_ramp, vmin=-1, vmax=1)
plt.colorbar(label='NDVI')
plt.title('Normalized Difference Vegetation Index (NDVI)')
plt.show()

output_filend = "E:/errors in qgis/LANDSAT2/forgif/moreNDVI.tif"

    # Write the EVI to a raster file with the same geospatial metadata as the input Landsat image
with rasterio.open(
    output_filend,
    'w',
    driver='GTiff',
     height=src.height,
    width=src.width,
    count=1,
    dtype=rasterio.float32,
    crs=src.crs,
    transform=src.transform,
    nodata=None
) as dst:
    dst.write(evi, 1)
    print("NDVI calculation completed and saved to:", output_filend)



# In[17]:


import rasterio 
import numpy as np
import matplotlib.pyplot as plt

# Open the Landsat image with 7 bands
with rasterio.open("E:/errors in qgis/LANDSAT2/2003.tif") as src:
    # Read the near-infrared (NIR) and red bands
    nir = src.read(4)
    red = src.read(3)

    # Calculate NDVI
    ndvi = (nir - red) / (nir + red)

    # Define the color ramp for NDVI visualization
    cmap = plt.cm.RdYlGn  # Red-Yellow-Green color ramp
    vmin, vmax = -1, 1     # NDVI ranges from -1 to 1

    # Plot the NDVI with the defined color ramp
    plt.figure(figsize=(10, 6))
    plt.imshow(ndvi, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(label='NDVI')
    plt.title('Normalized Difference Vegetation Index (NDVI)')
    plt.show()


# # EVI CALCULATIONS

# In[41]:


import rasterio 
import numpy as np
import matplotlib.pyplot as plt

# Open the Landsat image with 7 bands
with rasterio.open("E:/errors in qgis/LANDSAT2/2003.tif") as src:
    # Read the near-infrared (NIR), red, and blue bands
    nir = src.read(4)
    red = src.read(3)
    blue = src.read(1)

    # Calculate EVI
    evi = 2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1))

    # Define the color ramp for EVI visualization
    cmap = plt.cm.RdYlGn  # Red-Yellow-Green color ramp
    vmin, vmax = -2, 2     # EVI ranges from -2 to 2

    # Plot the EVI with the defined color ramp
    plt.figure(figsize=(10, 6))
    plt.imshow(evi, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(label='EVI')
    plt.title('Enhanced Vegetation Index (EVI)')
    plt.show()

output_file = "E:/errors in qgis/LANDSAT2/forgif/2003_EVI22.tif"

    # Write the EVI to a raster file with the same geospatial metadata as the input Landsat image
with rasterio.open(
    output_file,
    'w',
    driver='GTiff',
     height=src.height,
    width=src.width,
    count=1,
    dtype=rasterio.float32,
    crs=src.crs,
    transform=src.transform,
    nodata=None
) as dst:
    dst.write(evi, 1)
    print("EVI calculation completed and saved to:", output_file)




# In[31]:


import rasterio 
import numpy as np
import matplotlib.pyplot as plt

# Open the Landsat image with 7 bands
with rasterio.open("E:/errors in qgis/LANDSAT2/2003.tif") as src:
    # Read the near-infrared (NIR), red, and blue bands
    nir = src.read(4,masked=True)
    red = src.read(3,masked=True)
    blue = src.read(1,masked=True)

    # Calculate EVI
    evi = 2.5 * ((nir - red) / (nir + 6 * red - 7.5 * blue + 1))

    # Define the output file path
    output_file = "E:/errors in qgis/LANDSAT2/2003_EVI111.tif"

    # Write the EVI to a raster file with the same geospatial metadata as the input Landsat image
    with rasterio.open(
        output_file,
        'w',
        driver='GTiff',
        height=src.height,
        width=src.width,
        count=1,
        dtype=rasterio.float32,
        crs=src.crs,
        transform=src.transform,
        nodata=None
    ) as dst:
        dst.write(evi, 1)

    print("EVI calculation completed and saved to:", output_file)


# # more

# In[30]:


import rasterio 
import numpy as np

# Open the Landsat image with 7 bands
with rasterio.open("E:/errors in qgis/LANDSAT2/2003.tif") as src:
    # Read the near-infrared (NIR), red, and blue bands
    nir = src.read(4, masked=True)
    red = src.read(3,masked=True)
    blue = src.read(1,masked=True)

    # Calculate EVI, handling division by zero cases
    denominator = nir + 6 * red - 7.5 * blue + 1
    evi = np.where(denominator != 0, 2.5 * ((nir - red) / denominator), np.nan)

    # Get metadata from the source file
    meta = src.meta

    # Update metadata for the EVI file
    meta.update(dtype=rasterio.float32)

    # Specify the output file name
    output_filename = "E:/errors in qgis/LANDSAT2/2003_evi3.tif"

    # Write the EVI to a new GeoTIFF file
    with rasterio.open(output_filename, 'w', **meta) as dst:
        # Convert EVI array to float32 before writing
        evi_float32 = evi.astype(rasterio.float32)
        dst.write(evi_float32, 1)


# ## VISUALS PNGS AND GIF

# In[36]:


import rasterio 
import numpy as np
import matplotlib.pyplot as plt
import imageio

# Function to normalize array to range [0, 255]
def normalize_array(arr):
    min_val = np.nanmin(arr)
    max_val = np.nanmax(arr)
    return ((arr - min_val) / (max_val - min_val) * 255).astype(np.uint8)

# Open the Landsat image with 7 bands
with rasterio.open("E:/errors in qgis/LANDSAT2/2003.tif") as src:
    # Read the near-infrared (NIR), red, and blue bands
    nir = src.read(4, masked=True)
    red = src.read(3, masked=True)
    blue = src.read(1, masked=True)

    # Calculate NDVI
    ndvi = (nir - red) / (nir + red)
    
    # Calculate EVI, handling division by zero cases
    denominator = nir + 6 * red - 7.5 * blue + 1
    evi = np.where(denominator != 0, 2.5 * ((nir - red) / denominator), np.nan)

    # Normalize NDVI and EVI arrays to [0, 255] range
    ndvi_normalized = normalize_array(ndvi)
    evi_normalized = normalize_array(evi)

    # Save NDVI and EVI as PNG images
    plt.imsave("ndvi.png", ndvi_normalized, cmap='viridis')
    plt.imsave("evi.png", evi_normalized, cmap='RdYlGn')

# Create GIF file from PNG images
image_files = ["ndvi.png", "evi.png"]
images = [imageio.imread(file) for file in image_files]
imageio.mimsave("ndvi_evi.gif", images)



# In[37]:


# Save NDVI and EVI as PNG images to a specific output directory
plt.imsave("E:/errors in qgis/LANDSAT2/lfiles/ndvi.png", ndvi_normalized, cmap='viridis')
plt.imsave("E:/errors in qgis/LANDSAT2/lfiles/evi.png", evi_normalized, cmap='RdYlGn')

# Create GIF file from PNG images in a specific output directory
image_files = ["E:/errors in qgis/LANDSAT2/lfiles/ndvi.png", "E:/errors in qgis/LANDSAT2/lfiles/evi.png"]
imageio.mimsave("E:/errors in qgis/LANDSAT2/lfiles/ndvi_evi.gif", images)


# # gif files

# In[46]:


import imageio
from pathlib import Path

# Path to the directory containing TIFF images
image_path = Path('E:/errors in qgis/LANDSAT2/forgif/')

# List all TIFF images in the directory
images = list(image_path.glob('*.tif'))

# Create an empty list to store image data
image_list = []

# Loop through each TIFF image, read its content, and append to the list
for file_name in images:
    image_list.append(imageio.imread(file_name))

# Save the GIF file using imageio.mimsave
output_gif_path = 'E:/errors in qgis/LANDSAT2/forgif/image.gif'
imageio.mimsave(output_gif_path, image_list, fps=2)


# ![image.gif](attachment:8afbd9af-06d1-4937-8b6a-5df634a98037.gif)

# In[47]:


import imageio.v2 as imageio  # Importing the deprecated version to suppress the warning
from pathlib import Path

# Path to the directory containing TIFF images
image_path = Path('E:/errors in qgis/LANDSAT2/forgif/')

# List all TIFF images in the directory
images = list(image_path.glob('*.tif'))

# Create an empty list to store image data
image_list = []

# Loop through each TIFF image, read its content, and append to the list
for file_name in images:
    image_list.append(imageio.imread(file_name))

# Save the GIF file using imageio.mimsave
output_gif_path = 'E:/errors in qgis/LANDSAT2/forgif/image.gif'
imageio.mimsave(output_gif_path, image_list, fps=2)


# In[48]:


from IPython.display import Image



# Path to the GIF file

gif_path = 'E:/errors in qgis/LANDSAT2/forgif/image.gif'



# Display the GIF file

Image(filename=gif_path)



# In[45]:





# ![ndvi_evi.gif](attachment:fb0de07f-9b1a-4734-8dbd-ca4d717fc175.gif)

# In[ ]:




