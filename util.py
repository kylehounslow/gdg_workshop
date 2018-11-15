import os
import re
import glob
import logging
import requests
import numpy as np
import cv2
from PIL import Image
from tensorflow.python.lib.io import file_io

LOGGER = logging.getLogger(__name__)


def is_downloadable(url):
    """
    Does the url contain a downloadable resource
    """
    h = requests.head(url, allow_redirects=True)
    header = h.headers
    content_type = header.get('content-type')
    if 'text' in content_type.lower():
        return False
    if 'html' in content_type.lower():
        return False
    return True


def get_filename_from_cd(cd):
    """
    Get filename from content-disposition
    """
    if not cd:
        return None
    fname = re.findall('filename=(.+)', cd)
    if len(fname) == 0:
        return None
    return fname[0]


def download_image(url, save_to=None):
    """
    Download an image from URL
    Raises:
        Exceptions for file not downloadable, error file write, error file read.
    Args:
        url: valid image url
        save_to: (optional) save image to this path
    Returns:
        numpy.ndarray: Image as numpy array
    """
    if not is_downloadable(url=url):
        msg = 'url {} not downloadable'.format(url)
        LOGGER.error(msg)
        raise Exception(msg)
    req = requests.get(url, allow_redirects=True)
    filename = get_filename_from_cd(req.headers.get('content-disposition'))
    if filename is None:
        _, ext = os.path.splitext(url)
        filename = 'default{}'.format(ext)
    if save_to is not None:
        if not os.path.exists(save_to):
            os.makedirs(save_to)
        filepath = os.path.join(save_to, filename)
    else:
        filepath = filename
    with open(filepath, 'wb') as f:
        f.write(req.content)
    if not os.path.exists(filepath):
        msg = 'error writing {} to disk.'.format(filepath)
        LOGGER.error(msg)
        raise Exception(msg)
    img = np.array(Image.open(filepath))
    if img is None:
        msg = 'error reading {} from disk.'.format(filepath)
        LOGGER.error(msg)
        raise Exception(msg)
    if save_to is None:
        os.remove(filename)
    return img


def is_gcs_location(filename: str) -> bool:
    """
    Check whether filepath is to a Google Cloud Storage bucket
    Args:
        filename:

    Returns:
        bool: whether file is located in google cloud storage
    """
    # TODO: there may exist a better way to do this
    return filename.startswith("gs://") or filename.startswith("s3://")


def get_image_filenames(dirname):
    """
    Retrieve list of image filenames from directory or GCS bucket in order
    Args:
        dirname: path to directory or GCS bucket containing training data

    Returns:

    """
    if not isinstance(dirname, list):  # let user specify list of dirs
        dirname = [dirname]
    image_extensions = ['jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff']
    filenames = []
    for dir in dirname:
        for ext in image_extensions:
            if is_gcs_location(dir):
                files = file_io.get_matching_files(os.path.join(dir, '*.{}'.format(ext)))
            else:
                files = glob.glob(os.path.join(dir, '*.{}'.format(ext)))
            filenames.extend(files)
    return filenames


def frames_to_video(input_dir: str,
                    output_file: str,
                    fps: float,
                    resize: tuple = None):
    """
    convert directory with images to video
    Args:
        input_dir:
        output_file:
        fps:
        resize:

    Returns:

    Raises:
        Exception: No images found in directory

    """
    output_filename, output_ext = os.path.splitext(output_file)
    fourcc_str = None
    if output_ext.lower() == '.mkv':
        fourcc_str = 'XVID'
    input_filenames = get_image_filenames(input_dir)

    def _sorted_nicely(l: list):
        """ Sorts the given iterable in the way that is expected.

        Required arguments:
        l -- The iterable to be sorted.

        """
        import re
        convert = lambda text: int(text) if text.isdigit() else text
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        return sorted(l, key=alphanum_key)

    input_filenames = _sorted_nicely(input_filenames)
    if not input_filenames:
        msg = 'No images found in directory {}'.format(input_dir)
        LOGGER.error(msg)
        raise Exception(msg)

    def _load_img(filename: str):
        if is_gcs_location(filename):
            img = load_image_from_gcs(filename)
        else:
            img = cv2.imread(filename, cv2.IMREAD_COLOR)
        return img

    if resize:
        output_width, output_height = resize
    else:
        tmpfile = input_filenames[0]
        tmpimg = _load_img(tmpfile)
        output_width, output_height = tmpimg.shape[1], tmpimg.shape[0]
    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
    vw = cv2.VideoWriter(output_file, fourcc, fps, (int(output_width), int(output_height)))
    for input_filename in tqdm(input_filenames):
        img = _load_img(filename=input_filename)
        vw.write(img)
    vw.release()
