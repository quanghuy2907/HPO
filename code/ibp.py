import numpy as np
import cv2
from scipy import ndimage

import edi as edi_py



def calculate_shift(ref_img, images, mode='ecc'):
    shifts = []
    
    if mode == 'ecc':
        warp_mode = cv2.MOTION_TRANSLATION
        # Terminate the optimizer if it runs 50 iterations or error is small
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-10)

        print("Registering images (measuring random shifts) using ECC...")
        for i, img in enumerate(images):
            if i == 0:
                # The first image is ref; no shift.
                shifts.append([0., 0.])
                continue

            warp_matrix = np.eye(2, 3, dtype=np.float32)
            try:
                (cc, warp_matrix) = cv2.findTransformECC(ref_img, img, warp_matrix, warp_mode, criteria)
                # Matrix structure for Translation: [[1, 0, dx], [0, 1, dy]]
                dx = warp_matrix[0, 2]
                dy = warp_matrix[1, 2]
                shifts.append([dx, dy])
                print(f"Image {i} Shift: x={warp_matrix[0,2]:.4f}, y={warp_matrix[1,2]:.4f}")
                
            except cv2.error:
                print(f"Warning: Could not register image {i}")

    elif mode =='phase':
        ref_img = ref_img.astype(np.float32)
        h, w = ref_img.shape
        
        # Create a Hanning Window
        hann_window = cv2.createHanningWindow((w, h), cv2.CV_32F)
        
        
        print(f"Aligning {len(images)} images using Phase Correlation...")
        for i in range(0, len(images)):
            if i == 0:
                # The first image is ref; no shift.
                shifts.append([0., 0.])
                continue
            
            curr_img = images[i].astype(np.float32)
            shift, response = cv2.phaseCorrelate(ref_img, curr_img, window=hann_window)
            dx, dy = shift
            shifts.append((dx, dy))
            
            print(f"Image {i}: Shift={shift}, Confidence={response:.4f}")
    else: 
        raise(ValueError('unknown mode for shift calculation'))

    return shifts




def apply_tv_regularization(img, weight=0.02):
    """
    Calculates the Total Variation gradient to suppress noise.
    Imagine this as a 'smart smoothing' that respects edges.
    """
    # Calculate gradients (difference between adjacent pixels)
    diff_x = np.diff(img, axis=1, append=img[:, -1:])
    diff_y = np.diff(img, axis=0, append=img[-1:, :])

    # Sign of the gradient helps push values toward a smoother state
    # This is a simplified implementation of TV Gradient Descent
    grad_x = np.sign(diff_x)
    grad_y = np.sign(diff_y)

    # Shift back to align with pixels
    grad_x_back = np.roll(grad_x, 1, axis=1)
    grad_x_back[:, 0] = 0
    grad_y_back = np.roll(grad_y, 1, axis=0)
    grad_y_back[0, :] = 0

    # The divergence of the gradient field
    divergence = (grad_x - grad_x_back) + (grad_y - grad_y_back)
    
    return weight * divergence






def multi_frame_ibp(images_lr, shifts, scale_factor=2, iterations=30, blur_sigma=1.0, tv_weight=0.1):
    """
    Args:
        images_lr: List of low-res input images.
        shifts: List of (dx, dy) tuples representing sub-pixel motion for each image.
        scale_factor: Upscaling amount (e.g., 2 or 4).
        tv_weight: Strength of noise suppression (0.0 = no suppression).
    """
    h_lr, w_lr = images_lr[0].shape
    h_hr, w_hr = int(h_lr * scale_factor), int(w_lr * scale_factor)
    
    # initial HR, use the first image as reference
    img_hr_guess = cv2.resize(images_lr[0], (w_hr, h_hr), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    # img_hr_guess = ndimage.zoom(images_lr[0], (2, 2), order=3).astype(np.float32)
    # img_hr_guess = edi_py.EDI_upscale(images_lr[0], 4).astype(np.float32)

    # Pre-calculate Gaussian Kernel for blurring (Optics simulation)
    k_size = int(6 * blur_sigma) | 1
    gaussian_kernel = cv2.getGaussianKernel(k_size, blur_sigma)
    gaussian_kernel = gaussian_kernel * gaussian_kernel.T

    print(f"Starting IBP with {len(images_lr)} frames...")

    for i in range(iterations):
        total_error = np.zeros_like(img_hr_guess)
        
        # Loop through all 4 frames
        for idx, real_lr in enumerate(images_lr):
            dx, dy = shifts[idx]
            dx = -dx
            dy = -dy
            
            # --- A. SIMULATE THE CAMERA PROCESS ---
            # 1. Shift the HR Guess to match this specific frame
            # (We multiply shift by scale_factor because we are in HR space)
            M_shift = np.float32([[1, 0, -dx * scale_factor], [0, 1, -dy * scale_factor]]) 
            shifted_guess = cv2.warpAffine(img_hr_guess, M_shift, (w_hr, h_hr))
            # 2. Apply Optical Blur (PSF)
            blurred_guess = cv2.filter2D(shifted_guess, -1, gaussian_kernel)
            # 3. Downsample to Low Res
            simulated_lr = cv2.resize(blurred_guess, (w_lr, h_lr), interpolation=cv2.INTER_AREA)
            
            # --- B. CALCULATE ERROR ---
            diff = real_lr.astype(np.float32) - simulated_lr
    
            # --- C. BACK PROJECT ERROR ---
            # 1. Upsample the error map
            error_upscaled = cv2.resize(diff, (w_hr, h_hr), interpolation=cv2.INTER_CUBIC)
            # error_upscaled = cv2.filter2D(error_upscaled, -1, gaussian_kernel)
            # 2. Shift the error BACK to the reference position
            # Note the signs on dx/dy are reversed here to "undo" the shift
            M_unshift = np.float32([[1, 0, dx * scale_factor], [0, 1, dy * scale_factor]]) 
            error_aligned = cv2.warpAffine(error_upscaled, M_unshift, (w_hr, h_hr))
            
            # Accumulate error from this frame
            total_error += error_aligned

        # Average the error across all frames
        avg_error = total_error / len(images_lr)
        
        # --- D. UPDATE & REGULARIZE ---
        # Calculate TV Regularization term (Noise suppression)
        tv_term = apply_tv_regularization(img_hr_guess, weight=tv_weight)
        # Update equation: Guess + ErrorCorrection + SmoothnessConstraint
        img_hr_guess += avg_error + tv_term

    # Clip to valid range
    return np.clip(img_hr_guess, 0, 255).astype(np.uint8)