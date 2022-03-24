from typing import Dict, Any
from utils import get_gradients, to_grayscale
import numpy as np

NDArray = Any


def calc_insert_energy_matrices(I_gs):
    zero_column = np.broadcast_to([0.], [I_gs.shape[0], 1])
    zero_row = np.broadcast_to([0.], [1, I_gs.shape[1]])

    image_j_minus = np.concatenate([zero_column, I_gs[:, 0:-1]], axis=1)
    image_j_plus = np.concatenate([I_gs[:, 1:], zero_column], axis=1)

    image_i_minus = np.concatenate([zero_row, I_gs[0:-1]], axis=0)

    CV = np.abs(image_j_plus - image_j_minus)
    CL = CV + np.abs(image_i_minus - image_j_minus)
    CR = CV + np.abs(image_i_minus - image_j_plus)
    return CL, CV, CR


def resize(image: NDArray, out_height: int, out_width: int, forward_implementation: bool) -> Dict[str, NDArray]:
    """

    :param image: ِnp.array which represents an image.
    :param out_height: the resized image height
    :param out_width: the resized image width
    :param forward_implementation: a boolean flag that indicates whether forward or basic implementation is used.
                                    if forward_implementation is true then the forward-looking energy is used otherwise
                                    the basic implementation is used.
    :return: A dictionary with three elements, {'resized' : img1, 'vertical_seams' : img2 ,'horizontal_seams' : img3},
            where img1 is the resized image and img2/img3 are the visualization images
            (where the chosen seams are colored red and black for vertical and horizontal seams, respecitvely).
    """
    for i in range(abs(out_width - image.shape[1])):
        I_gs = to_grayscale(image)
        E = get_gradients(I_gs)
        M, indices, min_paths = calc_cost_matrix(E, I_gs, forward_implementation)
        seam = calc_seam(M, indices, min_paths)
        # TODO: remove seam from image
        # TODO: find out how to add / remove seam to resize image

    raise NotImplementedError('You need to implement this!')
    # TODO: return { 'resized' : img1, 'vertical_seams' : img2 ,'horizontal_seams' : img3}


def calc_seam(M, indices, min_paths):
    seam = np.empty(M.shape[0], dtype=object)
    j = np.argmin(M[-1])
    seam[-1] = (M.shape[0] - 1, j)
    for i in reversed(range(1, M.shape[0])):
        seam[i - 1] = indices[i - 1, j - 1 + min_paths[i, j]]
        j = j - 1 + min_paths[i, j]
    print("test")

    return seam


def calc_cost_matrix(E: NDArray, I_gs: NDArray, forward_implementation: bool):
    M = np.zeros_like(E)
    M_min_path = np.zeros_like(E, dtype=int)
    Indices_M = np.empty_like(E, dtype=object)
    CL, CV, CR = calc_insert_energy_matrices(I_gs)
    CV[:, 0] = 255.0

    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            M[i, j] = E[i, j]

            if i != 0 and i != (M.shape[0] - 1) and j != 0 and j != (M.shape[1] - 1):
                m1 = M[i - 1, j - 1] + CL[i, j]
                m2 = M[i - 1, j] + CV[i, j]
                m3 = M[i - 1, j + 1] + CR[i, j]
                vals = [m1, m2, m3]
                M[i, j] += min(vals)
                M_min_path[i, j] = np.argmin(vals)

            elif j == 0 and i != 0:
                m2 = M[i - 1, j] + CV[i, j]
                m3 = M[i - 1, j + 1] + CR[i, j]
                vals = [m2, m3]
                M[i, j] += min(vals)
                M_min_path[i, j] = np.argmin(vals)

            elif j == M.shape[1] - 1 and i != 0:
                m1 = M[i - 1, j - 1] + CL[i, j]
                m2 = M[i - 1, j] + CV[i, j]
                vals = [m1, m2]
                M[i, j] += min(vals)
                M_min_path[i, j] = np.argmin(vals)

            Indices_M[i, j] = (i, j)

    return M, Indices_M, M_min_path
