import fiona
import rasterio
import rasterio.mask
import os

# From https://rasterio.readthedocs.io/en/latest/topics/masking-by-shapefile.html
def clip_raster(raster_path, vector_path, output_path):
    with fiona.open(vector_path, "r") as shapefile:
        shapes = [feature["geometry"] for feature in shapefile]

    with rasterio.open(raster_path) as src:
        out_image, out_transform = rasterio.mask.mask(src, shapes, crop=True)
        out_meta = src.meta

    out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform})

    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(out_image)


def load_clips(clip_dir):
    clips = {}
    clip_files = os.listdir(clip_dir)
    for clip_file in clip_files:
        clip_path = os.path.join(clip_dir, clip_file)
        clip_name = clip_file.split('_')[0]
        clips[clip_name] = clip_path
    return clips

def load_inputs(input_dir):
    inputs = []
    input_files = os.listdir(input_dir)
    for input_file in input_files:
        input_path = os.path.join(input_dir, input_file)
        input_name = input_file.split('_')[0]
        inputs.append((input_name, input_file, input_path))
    return inputs

def clip_inputs(input_dir, clip_dir, output_dir):
    clips = load_clips(clip_dir)
    inputs = load_inputs(input_dir)

    print(f"clipping {len(inputs)} inputs to {output_dir}")

    for input_name, input_file, input_path in inputs:
        clip_path = clips[input_name]
        output_path = os.path.join(output_dir, input_file)
        print(f"clipping {input_path} to {output_path} using {clip_path}")
        clip_raster(input_path, clip_path, output_path)


clip_inputs("Input", "Clips", "Clipped-Input")

