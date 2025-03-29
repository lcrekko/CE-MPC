"""
utils.py

This module contains some useful small functions that
are helpful to the main purpose of the project.

1. random fixed-length vector generator (useful in MonteCarlo simulation)
2. random fixed-length vector generator for variable lengths (useful in MonteCarlo simulation)
3. color generator, generate a bunch of good colors


"""
from typing import Sequence
import numpy as np
from numpy.typing import NDArray

def generate_uniform_sphere_vectors(dim_vec: int, len_vec: float, num_vec: int,
                                    seed: int = 42) -> NDArray[np.float64]:
    """
    Generate N random l-dimensional vectors uniformly distributed on the sphere of radius r.

    Parameters:
    - dim_vec (int): Dimension of each vector.
    - len_vec (float): Desired norm of each vector.
    - num_vec (int): Number of vectors to generate.
    - seed (int, optional): Random seed for reproducibility. Default is 42.

    Returns:
    - NDArray[np.float64]: A (num_vec, dim_vec) array where
    each row is an dim_vec-dimensional vector with norm r.
    """
    # Initialize random number generator
    rng = np.random.default_rng(seed)

    # Generate num_vec random num_vec-dimensional vectors
    matrix_output = rng.standard_normal((num_vec, dim_vec))

    # Normalize to unit norm
    matrix_output /= np.linalg.norm(matrix_output, axis=1, keepdims=True)

    # Scale by the norm
    matrix_output *= len_vec
    return matrix_output

def generate_multiple_sphere_vectors(dim_vec: int, norms: Sequence[float], num_vec: int,
                                     seed: int = 42) -> NDArray[np.float64]:
    """
    Generate N random l-dimensional vectors for each norm in a given sequence,
    and return a (N, l, len(norms)) array.

    Parameters:
    - dim_vec (int): Dimension of each vector.
    - norms (Sequence[float]): A sequence of norms (r_1, r_2, ...).
    - num_vec (int): Number of vectors per norm.
    - seed (int, optional): Random seed for reproducibility. Default is 42.

    Returns:
    - NDArray[np.float64]: A (N, l, len(norms)) array where each slice along the last dimension
      corresponds to a different norm.
    """

    # Create the list of vectors for each norm
    vectors = [
        generate_uniform_sphere_vectors(dim_vec, r, num_vec, seed + i)
        for i, r in enumerate(norms)
    ]

    # Stack the vectors along the last axis
    return np.stack(vectors, axis=2)

def default_color_generator():
    """
    This function returns the default colors used in matplotlib
    :return:
    """
    my_color_dict = {'C0': (0.12156862745098039, 0.4666666666666667, 0.7058823529411765),
                     'C1': (1.0, 0.4980392156862745, 0.054901960784313725),
                     'C2': (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
                     'C3': (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
                     'C4': (0.5803921568627451, 0.403921568627451, 0.7411764705882353),
                     'C5': (0.5490196078431373, 0.33725490196078434, 0.29411764705882354),
                     'C6': (0.8901960784313725, 0.4666666666666667, 0.7607843137254902),
                     'C7': (0.4980392156862745, 0.4980392156862745, 0.4980392156862745),
                     'C8': (0.7372549019607844, 0.7411764705882353, 0.13333333333333333),
                     'C9': (0.09019607843137255, 0.7450980392156863, 0.8117647058823529)}

    return my_color_dict