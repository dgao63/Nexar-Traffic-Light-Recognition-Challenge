import numpy as np
import cv2
from scipy.ndimage import rotate

def augment_brightness(image, threshold=0.5):
    #gamma correction:
    prob = np.random.uniform()
    if prob > threshold:
        gamma = np.random.uniform(0.5, 1.5)
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                for i in np.arange(0, 256)]).astype("uint8")
        image = cv2.LUT(image, table)
    return image

def translation(image, x_shift_max=60, y_shift_max=30, threshold=0.5):
    # Translation and fill the holes with nearest pixels
    prob = np.random.uniform()
    if prob > threshold:
        x_shift = int(x_shift_max*np.random.uniform() - x_shift_max / 2)
        y_shift = int(y_shift_max*np.random.uniform() - y_shift_max / 2)
        M = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
        rows,cols = image.shape[0:2]
        image = cv2.warpAffine(image, M, (cols,rows))
        mask=np.zeros((rows,cols,1), np.uint8)
        if x_shift >= 0 and y_shift >= 0:
            mask[0:y_shift,:]=1
            mask[:,0:x_shift]=1
        elif x_shift >= 0 and y_shift < 0:
            mask[rows+y_shift:rows,:]=1
            mask[:,0:x_shift]=1
        elif x_shift < 0 and y_shift >= 0:
            mask[0:y_shift,:]=1
            mask[:,cols+x_shift:cols]=1
        else:
            mask[rows+y_shift:rows,:]=1
            mask[:,cols+x_shift:cols]=1
        image = cv2.inpaint(image,mask,3,cv2.INPAINT_TELEA)
    return image

def flip(image, threshold=0.5):
    prob = np.random.uniform()
    if prob > threshold:
        image = cv2.flip(image, 1)
    return image

def rotation(image, threshold=0.5):
    #rotate towards left if rot is positive
    prob = np.random.uniform()
    if prob>threshold:
        angles = [90, 270]
        rot = np.random.choice(angles)
        image = rotate(image, rot, reshape=False)
    return image

def shadow_mask(image, threshold=0.5):
    prob = np.random.uniform()
    if prob > threshold:
        rows,cols = image.shape[0:2]
        top_y = rows*np.random.uniform()
        top_x = cols*np.random.uniform()
        bot_x = cols*np.random.uniform()
        bot_y = rows*np.random.uniform()
        image_hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
        shadow_mask = 0*image_hls[:,:,1]
        X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
        Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]
        shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
            #random_bright = .25+.7*np.random.uniform()
        if np.random.randint(2)==1:
            random_bright = 0.4
            cond1 = shadow_mask==1
            cond0 = shadow_mask==0
            if np.random.randint(2)==1:
                image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
            else:
                image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright    
        image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)
    return image

def random_shear(image, shear_range=30, threshold=0.5):
    prob = np.random.uniform()
    if prob > threshold:    
        rows, cols, ch = image.shape
        dx = np.random.randint(-shear_range, shear_range + 1)
        random_point = [cols / 2 + dx, rows / 2]
        pts1 = np.float32([[0, rows], [cols, rows], [cols / 2, rows / 2]])
        pts2 = np.float32([[0, rows], [cols, rows], random_point])
        dsteering = dx / (rows / 2) * 360 / (2 * np.pi * 25.0) / 6.0
        M = cv2.getAffineTransform(pts1, pts2)
        image = cv2.warpAffine(image, M, (cols, rows), borderMode=1)
    return image
