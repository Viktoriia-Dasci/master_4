import os
import glob
import numpy as np
import nibabel as nib
import splitfolders

from pathlib import Path

# Define directories
input_folder = Path('/content/drive/MyDrive/Brats_2020_TrainData')
output_folder = Path('/content/drive/MyDrive/')

# Function to split folders
def split_folders_with_ratio(input_folder, output_folder, seed=42, ratio=(.70, .20, .10)):
    # List of folders to split
    folders = ['Masks', 'T1', 'T2', 'T1CE', 'Flair']

    # Loop over each folder and split
    for folder in folders:
        splitfolders.ratio(input_folder / folder, output=output_folder / f'{folder}_Split', seed=seed, ratio=ratio)

split_folders_with_ratio(input_folder, output_folder)

# List of file paths for training and testing
data_types = ['HGG', 'LGG']
modalities = ['flair', 't1', 't1ce', 't2', 'masks']
split_folders = ['train', 'val', 'test']

file_paths = {}
for modality in modalities:
    for data_type in data_types:
        for folder in split_folders:
            file_paths[f'{data_type}_{modality}_{folder}'] = sorted((output_folder / f'{modality}_split' / folder / f'{data_type}_{modality}').glob('*.nii'))

#Extracting and saving MRI slices for individual modalities
def extract_slices(mri_slices, masks_list, save_dir_img, save_dir_msk):
    for img in range(len(mri_slices)):
        slices = []
        slices_mask = []
        print("Now preparing image and masks number: ", img)

        # Load MRI slices
        temp_image = nib.load(mri_slices[img]).get_fdata()
        
        for n_slice in range(0, 155):
            MRslice = temp_image[:, :, n_slice]
            slices.append(MRslice)

        # Load masks
        temp_mask = nib.load(masks_list[img]).get_fdata()
        
        for n_slice in range(0, 155):
            MRslice_mask = temp_mask[:, :, n_slice]
            slices_mask.append(MRslice_mask)

        print(len(slices))
        print(len(slices_mask))

        for slice_n in range(len(slices)):
            val, counts = np.unique(slices_mask[slice_n].astype(np.uint8), return_counts=True)

            temp_image = slices[slice_n]
            masks = slices_mask[slice_n]

            if (1 - (counts[0] / counts.sum())) > 0.01:  # At least 1% useful volume with labels that are not 0
                print("Save Me")
                save_path_img = save_dir_img + '/' + str(slice_n) + '_' + str(img)
                np.save(save_path_img, temp_image)
                save_path_msk = save_dir_msk + '/' + str(slice_n) + '_' + str(img)
                np.save(save_path_msk, masks)
            else:
                print("I am useless") 


#Extracting and saving MRI slices and stacking 3 best modalities
def stack_3_slices(t2_list, t1ce_list, flair_list, masks_list, save_dir_img, save_dir_msk):
    for img in range(len(t2_list)):
        print("Now preparing image and masks number:", img)

        # Load T2 image
        t2_image = nib.load(t2_list[img]).get_fdata()
        
        # Load T1CE image
        t1ce_image = nib.load(t1ce_list[img]).get_fdata()
        
        # Load FLAIR image
        flair_image = nib.load(flair_list[img]).get_fdata()

        # Load mask
        temp_mask = nib.load(masks_list[img]).get_fdata()

        # Combine slices
        combined_slices = np.stack([t2_image, t1ce_image, flair_image], axis=3)

        for n_slice in range(combined_slices.shape[2]):
            MRslice = combined_slices[:, :, n_slice, :]
            MRslice_mask = temp_mask[:, :, n_slice]
            val, counts = np.unique(MRslice_mask.astype(np.uint8), return_counts=True)
            print(val)
            print(counts)

            if (1 - (counts[0] / counts.sum())) > 0.01:  # At least 1% useful volume with labels that are not 0
                print("Save Me")
                # Save the slice as an image
                save_path_img = save_dir_img + '/' + str(n_slice) + '_' + str(img)
                np.save(save_path_img, MRslice)
                
                # Save the corresponding mask
                save_path_msk = save_dir_msk + '/' + str(n_slice) + '_' + str(img)
                np.save(save_path_msk, MRslice_mask)

            else:
                print("I am useless")




#Extracting and saving MRI slices and stacking all 4 modalities
def stack_4_slices(t2_list, t1ce_list, flair_list, t1_list, masks_list, save_dir_img, save_dir_msk):

    for img in range(len(t2_list)):
        print("Now preparing image and masks number:", img)

        # Load T2 image
        t2_image = nib.load(t2_list[img]).get_fdata()
        
        # Load T1CE image
        t1ce_image = nib.load(t1ce_list[img]).get_fdata()
        
        # Load FLAIR image
        flair_image = nib.load(flair_list[img]).get_fdata()
        
        # Load T1 image
        t1_image = nib.load(t1_list[img]).get_fdata()

        # Load mask
        temp_mask = nib.load(masks_list[img]).get_fdata()

        # Combine slices
        combined_slices = np.stack([t2_image, t1ce_image, flair_image, t1_image], axis=3)

        for n_slice in range(combined_slices.shape[2]):
            MRslice = combined_slices[:, :, n_slice, :]
            MRslice_mask = temp_mask[:, :, n_slice]
            val, counts = np.unique(MRslice_mask.astype(np.uint8), return_counts=True)
            print(val)
            print(counts)

            if (1 - (counts[0] / counts.sum())) > 0.01:  # At least 1% useful volume with labels that are not 0
                print("Save Me")
                print(n_slice)
                # Save the slice as an image
                save_path_img = save_dir_img + '/' + str(n_slice) + '_' + str(img)
                np.save(save_path_img, MRslice)
                
                # Save the corresponding mask
                save_path_msk = save_dir_msk + '/' + str(n_slice) + '_' + str(img)
                np.save(save_path_msk, MRslice_mask)

            else:
                print("I am useless")

# Call the function for stacking 3 slices
for data_type in data_types:
    for folder in split_folders:
        stack_3_slices(file_paths[f'{data_type}_t2_{folder}'],
                       file_paths[f'{data_type}_t1ce_{folder}'], 
                       file_paths[f'{data_type}_flair_{folder}'], 
                       file_paths[f'{data_type}_masks_{folder}'], 
                       output_folder / f'Stacked_MRI_new' / folder / f'{data_type}_stack', 
                       output_folder / f'Stacked_Msk_new' / folder / f'{data_type}_masks')

# Call the function for stacking 4 slices
for data_type in data_types:
    for folder in split_folders:
        stack_4_slices(file_paths[f'{data_type}_t2_{folder}'], 
                       file_paths[f'{data_type}_t1ce_{folder}'], 
                       file_paths[f'{data_type}_flair_{folder}'], 
                       file_paths[f'{data_type}_t1_{folder}'], 
                       file_paths[f'{data_type}_masks_{folder}'], 
                       output_folder / f'Stacked_4_MRI_slices' / folder / f'{data_type}_stack', 
                       output_folder / f'Stacked_4_Msk_slices' / folder / f'{data_type}_masks')
