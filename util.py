import os
import re
import logging
import requests
import numpy as np
from PIL import Image

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


def download_image(url, save_to):
    """
    Download an image from URL
    Raises:
        Exceptions for file not downloadable, error file write, error file read.
    Args:
        url: valid image url
        save_to: save to this path
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
        filename = f'default{ext}'
    if not os.path.exists(save_to):
        os.makedirs(save_to)
    filepath = os.path.join(save_to, filename)
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
    return img, filepath