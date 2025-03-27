import pathlib

import matplotlib.pyplot as plt
import numpy as np


def save_fig(fig, png_file_path):
    '''
    Saves the given figure in the given file path.
    '''
    if type(png_file_path) is not pathlib.Path:
        png_file_path = pathlib.Path(png_file_path)
    print(f'Writing {png_file_path}')
    if not png_file_path.parent.is_dir():
        png_file_path.parent.mkdir(parents=True)
    fig.savefig(png_file_path)
    plt.close(fig.number)


def dates_filter(time, start_date, end_date):
    '''
    Returns the time indices of that are between the given dates.
    '''
    mask_1 = time >= start_date
    mask_2 = time <= end_date
    mask = mask_1*mask_2
    indices = np.where(mask)[0]
    return indices

