import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def show_mask_by_filename(filename, title=None):
    """
    Display a predicted mask image by filename.

    Parameters:
        filename (str): Full path or filename (from temp or static dir).
        title (str): Optional title to show above the image.
    """
    if not os.path.exists(filename):
        print(f"[Error] File not found: {filename}")
        return
    
    img = mpimg.imread(filename)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    if title:
        plt.title(title)
    plt.show()

show_mask_by_filename(r"G:\Projects\Cellula_Internship\Water_Bodies_Semantic_Segmentation\data\labels\23.png", title="True Mask")