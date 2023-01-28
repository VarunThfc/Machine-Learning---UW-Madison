import sys
import glob
import h5py
import cv2

IMG_WIDTH = 512
IMG_HEIGHT = 512

h5file = 'import_images.h5'

nfiles = len(glob.glob('./publaynet/train/*.jpg'))
print(f'count of image files nfiles={nfiles}')

# # resize all images and load into a single dataset
# with h5py.File(h5file,'w') as  h5f:
#     img_ds = h5f.create_dataset('images',shape=(nfiles, IMG_WIDTH, IMG_HEIGHT,3), dtype=int)
#     for cnt, ifile in enumerate(glob.iglob('./publaynet/train*.jpg')) :
#         img = cv2.imread(ifile, cv2.IMREAD_COLOR)
#         # or use cv2.IMREAD_GRAYSCALE, cv2.IMREAD_UNCHANGED
#         img_resize = cv2.resize( img, (IMG_WIDTH, IMG_HEIGHT) )
#         img_ds[cnt:cnt+1:,:,:] = img_resize
        
with h5py.File('nds_'+h5file,'w') as  h5f:
    for cnt, ifile in enumerate(glob.iglob('./publaynet/train/*.jpg')) :
        img = cv2.imread(ifile, cv2.IMREAD_COLOR)
        # or use cv2.IMREAD_GRAYSCALE, cv2.IMREAD_UNCHANGED
        img_ds = h5f.create_dataset('images_'+f'{cnt+1:03}', data=img)