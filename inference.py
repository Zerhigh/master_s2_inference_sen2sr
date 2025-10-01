import mlstac
import torch
from pathlib import Path
import pandas as pd
import rasterio
from rasterio.transform import Affine
from tqdm import tqdm
import os

base = Path('/data/databases/SAMUEL/s2_data/')
stratification = base / 'stratification_tables/full'
s2_images = base / 'lr_s2'
to_path = base / 'inferred_s2'

data = [pd.read_csv(file) for file in stratification.glob('*.csv')]
s2_data = pd.concat(data)


# Download the model
if not Path("samuel_inference/model/SEN2SR_RGBN").exists():
    mlstac.download(
    file="https://huggingface.co/tacofoundation/sen2sr/resolve/main/SEN2SR/NonReference_RGBN_x4/mlm.json",
    output_dir="samuel_inference/model/SEN2SR_RGBN",
    )

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = mlstac.load("samuel_inference/model/SEN2SR_RGBN").compiled_model(device=device)
model = model.to(device)

# Prepare the data to be used in the model, select just one sample 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for i, file in tqdm(enumerate(s2_data['lr_s2_path'])):
    # open sentinel2 tile
    filename = Path(file).name
    if Path(to_path / filename).exists():
        continue
    else:
        with rasterio.open(s2_images / filename) as src:
            # just get r, g, b, nir
            img = (src.read([4, 3, 2, 8]) / 10_000).astype('float32')

            X = torch.from_numpy(img).float().to(device)
            superX = model(X[None]).squeeze(0).cpu().numpy()
            #superX = superX / 10_000
            # print(superX)
            # print(img.shape, img.dtype)
            # print(superX.shape, superX.dtype)

            new_transform = Affine(2.5, src.transform.b, src.transform.c,
                            src.transform.d, -2.5, src.transform.f)

            # Convert the data array to NumPy and scale
            b, h, w = superX.shape

            with rasterio.open(
                    to_path / filename,
                    "w",
                    driver="GTiff",
                    height=h,
                    width=w,
                    count=b,
                    dtype=superX.dtype,
                    crs=src.crs,
                    transform=new_transform,
                    nodata=0,
                    compress="zstd",
                    zstd_level=13,
                    interleave="band",
            ) as dst:
                dst.write(superX)
            
