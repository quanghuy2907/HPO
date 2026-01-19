import cv2
import numpy as np
import matplotlib.pyplot as plt




def register_images(ref_img, images, mode='ecc'):
    homographies = []
    
    if mode == 'ecc':
        warp_mode = cv2.MOTION_TRANSLATION
        # Terminate the optimizer if it runs 50 iterations or error is small
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-10)

        print("Registering images (measuring random shifts) using ECC...")
        for i, img in enumerate(images):
            if i == 0:
                # The first image is ref; no shift.
                homographies.append(np.eye(2, 3, dtype=np.float32))
                continue
                
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            try:
                (cc, warp_matrix) = cv2.findTransformECC(ref_img, img, warp_matrix, warp_mode, criteria)
                homographies.append(warp_matrix)
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
                homographies.append(np.eye(2, 3, dtype=np.float32))
                continue

            curr_img = images[i].astype(np.float32)
            shift, response = cv2.phaseCorrelate(ref_img, curr_img, window=hann_window)
            dx, dy = shift
            homographies.append(np.array([[1., 0., dx], [0., 1., dy]]))
            
            print(f"Image {i}: Shift={shift}, Confidence={response:.4f}")
    else: 
        raise(ValueError('unknown mode for shift calculation'))

    return homographies




def super_resolve_drizzle(images, homographies, scale=2, sampling_method='nearest'):
    """
    Reconstructs High-Res image using Weighted Accumulation (Splatting).
    """
    h, w = images[0].shape
    hr_h, hr_w = h * scale, w * scale
    
    # 1. Accumulation Buffers 
    accumulator = np.zeros((hr_h, hr_w), dtype=np.float32)
    divisor = np.zeros((hr_h, hr_w), dtype=np.float32)
    
    print("Accumulating/Splatting pixels...")
    for idx, img in enumerate(images):
        # Get the shift for current image
        M = homographies[idx]
        shift_x = M[0, 2]
        shift_y = M[1, 2]
        
        # Scale the shift to the High-Res 
        # If image shifted 0.5 px in Low-Res, it shifts 1.0 px in High-Res (2x)
        M_hr = np.array([[1, 0, shift_x * scale],
                         [0, 1, shift_y * scale]], dtype=np.float32)
        
        # Upscale the Low-Res image to High-Res size (Nearest Neighbor)
        if sampling_method == 'nearest':
            img_hr_view = cv2.resize(img, (hr_w, hr_h), interpolation=cv2.INTER_NEAREST)
        else:
            raise(ValueError('Wrong method'))
        
        # Warp the image to align with the reference
        aligned_img = cv2.warpAffine(
            img_hr_view, M_hr, (hr_w, hr_h), 
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT, 
            borderValue=0
        )
        
        # Create a Weight Mask, simplified by giving 1 for any pxel having value
        mask = (aligned_img > 0).astype(np.float32)
        
        accumulator += aligned_img
        divisor += mask

    # Normalize (Average)
    divisor[divisor == 0] = 1 
    sr_image = accumulator / divisor
    
    return np.clip(sr_image, 0, 255).astype(np.uint8)



