
import torch
import math

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / abs(math.pow(10000, 2 * (hid_idx // 2) / d_hid))

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = torch.zeros(n_position, d_hid * 2)
    for position in range(n_position):
        angle_vec = get_posi_angle_vec(position)
        angle_rads = torch.tensor([[angle] * 2 for angle in angle_vec])  # 2 * d_hid
        sinusoid_table[position] = torch.cat([torch.sin(angle_rads[:, 0]), torch.cos(angle_rads[:, 1])])

    if padding_idx is not None:
        sinusoid_table[padding_idx] = 0

    return sinusoid_table

# Example usage
n_position = 100
d_hid = 10
sinusoid_table = get_sinusoid_encoding_table(n_position, d_hid)

sinusoid_table.shape