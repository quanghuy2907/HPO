import numpy as np
from scipy.signal import convolve2d


SOBEL_THRESHOLD = 30  # 50 in patent
UM_THRESHOLD = 10
UM_STRENGTH = 0.3
UM_THRESHOLD_CUT = 0.2

BLUR_KERNEL = np.array([
	[0.0947416, 0.118318, 0.0947416],
	[0.118318, 0.147761, 0.118318],
	[0.0947416, 0.118318, 0.0947416]
])

EDGE_KERNEL_X = np.array([
	[-1, 0, 1],
	[-2, 0, 2],
	[-1, 0, 1]
])

EDGE_KERNEL_Y = np.array([
	[-1, -2, -1],
	[0, 0, 0],
	[1, 2, 1]
])


def pad(img, pad_size):
	h, w = img.shape
	ph = h + pad_size * 2
	pw = w + pad_size * 2
	out = np.zeros((ph, pw), dtype=img.dtype)
	out[pad_size:pad_size + h, pad_size:pad_size + w] = img
	return out


def conv2d(img, kernel):
	out = np.zeros_like(img)
	height, width = img.shape
	kernel_size = kernel.shape[0]
	offset = kernel_size // 2
	for ch in range(offset, height - offset):
		for cw in range(offset, width - offset):
			window = img[ch - offset:ch + offset + 1, cw - offset:cw + offset + 1]
			out[ch, cw] = np.sum(window * kernel)
	return out


def blur(img):
	return conv2d(img, BLUR_KERNEL)


def detect_edge(img):
	edge_x = conv2d(img, EDGE_KERNEL_X)
	edge_y = conv2d(img, EDGE_KERNEL_Y)
	edge = np.sqrt(edge_x ** 2 + edge_y ** 2)
	edge = np.where(edge > SOBEL_THRESHOLD, 1., 0.)
	return edge


def upsample(img):
	h, w = img.shape
	out = np.zeros((h * 2, w * 2), dtype=img.dtype)
	out[::2, ::2] = img
	out[1::2, ::2] = img
	out[::2, 1::2] = img
	out[1::2, 1::2] = img
	return out


def initialize_mask(lr_pad):
	blurred_pad = blur(lr_pad)
	edge_pad = detect_edge(lr_pad)
	mask_pad = lr_pad - blurred_pad
	return mask_pad, edge_pad


def update_mask(mask_pad, edge_pad):
	height, width = mask_pad.shape
	for h in range(height):
		for w in range(width):
			if (edge_pad[h, w] == 0) and (mask_pad[h, w] <= UM_THRESHOLD_CUT):
				mask_pad[h, w] = 0
			else:
				mask_pad[h, w] = mask_pad[h, w] * UM_STRENGTH
	return mask_pad


def update_enhance(lr_pad, enhance_pad, mask_pad, edge_pad):
	kernel_size = 3
	offset = kernel_size // 2
	height, width = enhance_pad.shape
	for h in range(height):
		for w in range(width):
			if edge_pad[h, w] == 0:
				continue
			lr_val = lr_pad[h, w]
			enhance_val = enhance_pad[h, w]
			window = lr_pad[h - offset:h + offset + 1, w - offset:w + offset + 1]
			maxval = np.max(window)
			minval = np.min(window)
			if (lr_val == maxval) or (lr_val == minval):
				continue
			if (enhance_val <= maxval) or (enhance_val >= minval):
				continue
			mask_pad[h, w] = mask_pad[h, w] * UM_STRENGTH
			enhance_pad[h, w] = lr_pad[h, w] + mask_pad[h, w]
	return enhance_pad, mask_pad


def enhance_lr(lr):
	lr_pad = pad(lr, 1)
	mask_pad, edge_pad = initialize_mask(lr_pad)
	mask_pad = update_mask(mask_pad, edge_pad)
	enhance_pad = lr_pad + mask_pad
	enhance_pad, mask_pad = update_enhance(lr_pad, enhance_pad, mask_pad, edge_pad)
	enhance = enhance_pad[1:-1, 1:-1]
	mask = mask_pad[1:-1, 1:-1]
	return enhance, mask


def SR_temp(lr1, lr2, lr3, lr4):
	sr_tmp1 = upsample(lr1)
	sr_tmp2 = np.pad(upsample(lr2), ((0, 0), (1, 0)), "constant", constant_values=0)[:, :-1]
	sr_tmp3 = np.pad(upsample(lr3), ((1, 0), (0, 0)), "constant", constant_values=0)[:-1, :]
	sr_tmp4 = np.pad(upsample(lr4), ((1, 0), (1, 0)), "constant", constant_values=0)[:-1, :-1]
	sr_tmp = (sr_tmp1 + sr_tmp2 + sr_tmp3 + sr_tmp4) / 4
	return sr_tmp


def SR_edge_temp(edge1, edge2, edge3, edge4):
	sr_edge1 = upsample(edge1)
	sr_edge2 = np.pad(upsample(edge2), ((0, 0), (1, 0)), "constant", constant_values=0)[:, :-1]
	sr_edge3 = np.pad(upsample(edge3), ((1, 0), (0, 0)), "constant", constant_values=0)[:-1, :]
	sr_edge4 = np.pad(upsample(edge4), ((1, 0), (1, 0)), "constant", constant_values=0)[:-1, :-1]
	height, width = sr_edge1.shape
	sr_edge = np.zeros((height, width))
	for h in range(height):
		for w in range(width):
			e1 = sr_edge1[h, w]
			e2 = sr_edge2[h, w]
			e3 = sr_edge3[h, w]
			e4 = sr_edge4[h, w]
			if e1 > 0 and e2 > 0 and e3 > 0 and e4 > 0:
				sr_edge[h, w] = max(e1, e2, e3, e4)
			elif e1 < 0 and e2 < 0 and e3 < 0 and e4 < 0:
				sr_edge[h, w] = min(e1, e2, e3, e4)
			else:
				sr_edge[h, w] = (e1 + e2 + e3 + e4) / 4
	return sr_edge


def HPO_shuffle(LRs):
	LR1, LR2, LR3, LR4 = LRs
	HPO = np.zeros((LR1.shape[0] * 2, LR1.shape[1] * 2))
	HPO[0::2, 0::2] = LR1
	HPO[0::2, 1::2] = LR2
	HPO[1::2, 0::2] = LR3
	HPO[1::2, 1::2] = LR4
	return HPO


def HPO_average(HR):
	if HR.shape[0] % 2 == 0:
		HR = HR[:-1]
	if HR.shape[1] % 2 == 0:
		HR = HR[:, :-1]
	convolve_matrix = np.array([[1/16, 1/8, 1/16], [1/8, 1/4, 1/8], [1/16, 1/8, 1/16]])
	HPO_average_0 = convolve2d(HR, convolve_matrix, mode="valid")
	return HPO_average_0


def HPO_shuffle_edge(lrs, edge_coeff):
	lr1, lr2, lr3, lr4 = lrs
	enhance1, edge1 = enhance_lr(lr1)
	enhance2, edge2 = enhance_lr(lr2)
	enhance3, edge3 = enhance_lr(lr3)
	enhance4, edge4 = enhance_lr(lr4)
	sr_tmp = HPO_shuffle((enhance1, enhance2, enhance3, enhance4))
	sr_edge = HPO_shuffle((edge1, edge2, edge3, edge4))
	sr_0 = sr_tmp + edge_coeff * sr_edge
	return sr_0


def HPO_average_edge(lrs, edge_coeff):
	lr1, lr2, lr3, lr4 = lrs
	enhance1, edge1 = enhance_lr(lr1)
	enhance2, edge2 = enhance_lr(lr2)
	enhance3, edge3 = enhance_lr(lr3)
	enhance4, edge4 = enhance_lr(lr4)
	sr_tmp = SR_temp(enhance1, enhance2, enhance3, enhance4)
	sr_edge = SR_edge_temp(edge1, edge2, edge3, edge4)
	sr_0 = sr_tmp + edge_coeff * sr_edge
	return sr_0
