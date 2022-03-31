from typing import Dict, Any
from utils import get_gradients, to_grayscale
import numpy as np

NDArray = Any


def calc_insert_energy_matrices(I_gs, const_factor):
    zero_column = np.broadcast_to([0.], [I_gs.shape[0], 1])
    zero_row = np.broadcast_to([0.], [1, I_gs.shape[1]])

    image_j_minus = np.concatenate([zero_column, I_gs[:, 0:-1]], axis=1)  # (i,j-1)
    image_j_plus = np.concatenate([I_gs[:, 1:], zero_column], axis=1)  # (i, j+1)

    image_i_minus = np.concatenate([zero_row, I_gs[0:-1]], axis=0)  # (i-1, j)

    CV = np.abs(image_j_plus - image_j_minus)
    CV[:, 0] = const_factor
    CV[:, -1] = const_factor
    CL = CV + np.abs(image_i_minus - image_j_minus)
    CR = CV + np.abs(image_i_minus - image_j_plus)
    return CL, CV, CR


def enlarge_image(image, mask_mat, param, out_width):
    pass


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
    resized = image
    vertical_seams = image
    horizontal_seams = image
    if image.shape[1] - out_width != 0:
        resized, vertical_seams = resize_width(image, out_width, forward_implementation, [255, 0, 0])

    image = resized
    if image.shape[0] - out_height != 0:
        image = np.rot90(image, k=1)
        resized, horizontal_seams = resize_width(image, out_height, forward_implementation, [0, 0, 0])
        resized = np.rot90(resized, k=3)
        horizontal_seams = np.rot90(horizontal_seams, k=3)

    return {'resized': resized, 'vertical_seams': vertical_seams, 'horizontal_seams': horizontal_seams}

    # TODO: remove seam from image
    # TODO: find out how to add / remove seam to resize image

    # TODO: return { 'resized' : img1, 'vertical_seams' : img2 ,'horizontal_seams' : img3}


def resize_width(image, new_width, forward, color):
    mask_mat = np.full(shape=(image.shape[0], image.shape[1]), fill_value=True)
    W_copy = image.copy()
    I_gs = to_grayscale(image)
    indices = np.outer(np.ones(image.shape[0], dtype=int), np.arange(image.shape[1],dtype=int))
    for i in range(abs(new_width - image.shape[1])):
        E = get_gradients(I_gs)
        M, min_paths = calc_cost_matrix(E, I_gs, forward, 255)

        I_gs, W_copy = calc_seam(M, indices, min_paths, mask_mat, W_copy, I_gs, color)

    if image.shape[1] - new_width > 0:
        image = reduce_image(image, mask_mat, image.shape[0], new_width)
    else:
        image = enlarge_image(image, mask_mat, image.shape[0], new_width)

    return image, W_copy


def reduce_image(image, mask_mat, out_height, out_width):
    new_image = np.zeros((out_height, out_width, 3))
    for i in range(mask_mat.shape[0]):
        index = 0
        for j in range(mask_mat.shape[1]):
            if mask_mat[i, j]:
                new_image[i, index] = image[i, j]
                index += 1
    return new_image


def calc_seam(M, indices, min_paths, mask_mat, W_copy, I_gs, color):
    j = np.argmin(M[-1])
    mask_mat[-1, j] = False

    W_copy[-1, j] = color
    I_gs[-1, j:-1] = I_gs[-1, j + 1:]
    indices[-1, j:-1] = indices[-1, j + 1:]
    for i in reversed(range(1, M.shape[0])):
        row, col = i - 1, indices[i - 1, j - 1 + min_paths[i, j]]
        mask_mat[row, col] = False

        W_copy[row, col] = color
        I_gs[row, col:-1] = I_gs[row, col + 1:]
        indices[row, col:-1] = indices[row, col + 1:]
        j = j - 1 + min_paths[i, j]

    I_gs = I_gs[:, :-1]

    return I_gs, W_copy


def calc_cost_matrix(E: NDArray, I_gs: NDArray, forward_implementation: bool, const_factor: int):
    M = np.zeros_like(E)
    M_min_path = np.zeros_like(E, dtype=int)
    if forward_implementation:
        CL, CV, CR = calc_insert_energy_matrices(I_gs, const_factor)
    else:
        CL, CV, CR = np.zeros_like(E), np.zeros_like(E), np.zeros_like(E)

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

    return M, M_min_path
