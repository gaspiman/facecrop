from PIL import Image
import cv2
import logging
import argparse
from functools import partial
import sys
from autocrop import Cropper
from os import listdir
from os.path import isfile, join
from multiprocessing import Pool
import platform
import multiprocessing


def resize_and_crop(img_path, modified_path, size, crop_type='middle'):
    """
    Resize and crop an image to fit the specified size.
    args:
        img_path: path for the image to resize.
        modified_path: path to store the modified image.
        size: `(width, height)` tuple.
        crop_type: can be 'top', 'middle' or 'bottom', depending on this
            value, the image will cropped getting the 'top/left', 'midle' or
            'bottom/rigth' of the image to fit the size.
    raises:
        Exception: if can not open the file in img_path of there is problems
            to save the image.
        ValueError: if an invalid `crop_type` is provided.
    """
    # If height is higher we resize vertically, if not we resize horizontally
    img = Image.open(img_path)
    # Get current and desired ratio for the images
    img_ratio = img.size[0] / float(img.size[1])
    ratio = size[0] / float(size[1])
    # The image is scaled/cropped vertically or horizontally depending on the ratio
    if ratio > img_ratio:
        img = img.resize((size[0], int(size[0] * img.size[1] / img.size[0])),
                         Image.ANTIALIAS)
        # Crop in the top, middle or bottom
        if crop_type == 'top':
            box = (0, 0, img.size[0], size[1])
        elif crop_type == 'middle':
            box = (0, int((img.size[1] - size[1]) / 2),
                   img.size[0], int((img.size[1] + size[1]) / 2))
        elif crop_type == 'bottom':
            box = (0, img.size[1] - size[1], img.size[0], img.size[1])
        else:
            raise ValueError('ERROR: invalid value for crop_type')
        img = img.crop(box)
    elif ratio < img_ratio:
        img = img.resize((int(size[1] * img.size[0] / img.size[1]), size[1]),
                         Image.ANTIALIAS)
        # Crop in the top, middle or bottom
        if crop_type == 'top':
            box = (0, 0, size[0], img.size[1])
        elif crop_type == 'middle':
            box = (int((img.size[0] - size[0]) / 2), 0,
                   int((img.size[0] + size[0]) / 2), img.size[1])
        elif crop_type == 'bottom':
            box = (img.size[0] - size[0], 0, img.size[0], img.size[1])
        else:
            raise ValueError('ERROR: invalid value for crop_type')
        img = img.crop(box)
    else:
        img = img.resize((size[0], size[1]),
                         Image.ANTIALIAS)
        # If the scale is the same, we do not need to crop
    img.save(modified_path)


def facedetect(width, height, face_percent, inpath, outpath, prefix, f):
    cropper = Cropper(
        width=width,
        height=height,
        face_percent=face_percent,
    )
    print(join(inpath, f))
    infile = join(inpath, f)
    outfile = join(outpath, prefix+f)
    print("in:", infile, "out:", outfile)
    # Get a Numpy array of the cropped image
    try:
        cropped_array = cropper.crop(infile)
    except Exception as e:
        print(e)
        return

    if cropped_array is None:
        logging.info("No face detected. Continuing...")
        resize_and_crop(infile, outfile, (width, height))
        return

    # Save the cropped image with PIL
    try:
        cropped_image = Image.fromarray(cropped_array)
    except Exception as e:
        logging.info("couldn't resize:", e)
        return
    cropped_image.save(outfile)


if __name__ == "__main__":
    if platform.system() == "Darwin":
        multiprocessing.set_start_method('spawn')
    args = sys.argv[1:]
    parser = argparse.ArgumentParser(description="Parses command.")
    parser.add_argument("-width", "--width", type=int,
                        help="Output image width")
    parser.add_argument("-height", "--height", type=int,
                        help="Output image height")
    parser.add_argument("-fp", "--face_percent", type=int,
                        help="Face percent from original image")
    parser.add_argument("-i", "--input", help="Images input folder path")
    parser.add_argument("-o", "--output", help="Images output folder path")
    parser.add_argument("-pr", "--prefix", help="Prefix to all output image")

    options = parser.parse_args(args)
    print(options.width)

    onlyfiles = [f for f in listdir(
        options.input) if isfile(join(options.input, f))]

    pool = Pool(multiprocessing.cpu_count())
    func = partial(facedetect, options.width,
                   options.height,
                   options.face_percent,
                   options.input,
                   options.output,
                   options.prefix,
                   )

    pool.map(func, onlyfiles)
    pool.close()
    pool.join()

    """
    facedetect(
        options.width,
        options.height,
        options.face_percent,
        options.input,
        options.output,
        options.prefix,
    )
    """
