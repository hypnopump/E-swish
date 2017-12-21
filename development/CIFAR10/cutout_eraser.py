import numpy as np

def get_random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255, pixel_level=False):
    def eraser(input_img):
        img_h, img_w, img_c = input_img.shape
        p_1 = np.random.rand()

        if p_1 > p: return input_img

        while True:
            # Calculate the dimensions of the crop
            w = int(img_w * s_l)
            h = int(img_h * s_h)
            #Select starting coords
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)
            # If crop is in the image, cut out
            if left + w <= img_w and top + h <= img_h:
                break

        if pixel_level:
            c = np.random.uniform(v_l, v_h, (h, w, img_c))
        else:
            c = 0 # np.random.uniform(v_l, v_h)

        input_img[top:top + h, left:left + w, :] = c

        return input_img

    return eraser