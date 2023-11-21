import os

import pandas as pd

def import_cbis_dataset():
    csv_root = '/content/drive/MyDrive/archive/csv'
    img_root = '/content/drive/MyDrive/archive/jpeg'
    
    df_dicom = pd.read_csv(csv_root + '/' + 'dicom_info.csv')

    if df_dicom.empty:
        print('DataFrame is empty!')

    # print(df_dicom.head())
    cropped_images = df_dicom[df_dicom.SeriesDescription=='cropped images'].image_path
    # print(cropped_images.head(5))
    full_mammo = df_dicom[df_dicom.SeriesDescription=='full mammogram images'].image_path 
    # print(full_mammo.head(5))
    roi_img = df_dicom[df_dicom.SeriesDescription=='ROI mask images'].image_path
    # print(roi_img.head(5)) 

    cropped_images = cropped_images.replace('CBIS-DDSM/jpeg', img_root, regex=True)
    full_mammo = full_mammo.replace('CBIS-DDSM/jpeg', img_root, regex=True)
    roi_img = roi_img.replace('CBIS-DDSM/jpeg', img_root, regex=True)

    # view new paths
    # print('Cropped Images paths:\n')
    # print(cropped_images.iloc[0])
    # print('Full mammo Images paths:\n')
    # print(full_mammo.iloc[0])
    # print('ROI Mask Images paths:\n')
    # print(roi_img.iloc[0])

    # organize image paths
    full_mammo_dict = dict()
    cropped_images_dict = dict()
    roi_img_dict = dict()

    for dicom in full_mammo:
        key = dicom.split("/")[6]
        full_mammo_dict[key] = dicom
    for dicom in cropped_images:
        key = dicom.split("/")[6]
        cropped_images_dict[key] = dicom
    for dicom in roi_img:
        key = dicom.split("/")[6]
        roi_img_dict[key] = dicom

    # print(next(iter((full_mammo_dict.items()))))
    # load the mass dataset
    mass_train = pd.read_csv(csv_root + '/' + 'mass_case_description_train_set.csv')
    mass_test = pd.read_csv(csv_root + '/' + 'mass_case_description_test_set.csv')

    # print(mass_train.head())

    def fix_image_path(data):
        """correct dicom paths to correct image paths"""
        for index, img in enumerate(data.values):
            img_name = img[11].split("/")[2]
            data.iloc[index,11] = full_mammo_dict[img_name]
            img_name = img[12].split("/")[2]
            data.iloc[index,12] = cropped_images_dict[img_name]
        
    # apply to datasets
    fix_image_path(mass_train)
    fix_image_path(mass_test)

    # check unique values in pathology column
    # print(mass_train.pathology.unique())

    mass_train = mass_train.rename(columns={'left or right breast': 'left_or_right_breast',
                                           'image view': 'image_view',
                                           'abnormality id': 'abnormality_id',
                                           'abnormality type': 'abnormality_type',
                                           'mass shape': 'mass_shape',
                                           'mass margins': 'mass_margins',
                                           'image file path': 'image_file_path',
                                           'cropped image file path': 'cropped_image_file_path',
                                           'ROI mask file path': 'ROI_mask_file_path'})

    # print(mass_train.head(5))

    # fill in missing values using the backwards fill method
    mass_train['mass_shape'] = mass_train['mass_shape'].fillna(method='bfill')
    mass_train['mass_margins'] = mass_train['mass_margins'].fillna(method='bfill')

    #check null values
    # print(mass_train.isnull().sum())

    print(f'Shape of mass_train: {mass_train.shape}')
    print(f'Shape of mass_test: {mass_test.shape}')

    mass_test = mass_test.rename(columns={'left or right breast': 'left_or_right_breast',
                                           'image view': 'image_view',
                                           'abnormality id': 'abnormality_id',
                                           'abnormality type': 'abnormality_type',
                                           'mass shape': 'mass_shape',
                                           'mass margins': 'mass_margins',
                                           'image file path': 'image_file_path',
                                           'cropped image file path': 'cropped_image_file_path',
                                           'ROI mask file path': 'ROI_mask_file_path'})

    # view renamed columns
    # print(mass_test.columns)

    # fill in missing values using the backwards fill method
    mass_test['mass_margins'] = mass_test['mass_margins'].fillna(method='bfill')

    #check null values
    # print(mass_test.isnull().sum())

    full_mass = pd.concat([mass_train, mass_test], axis=0)
    print("Returning data from dataset_preparation method")
    # print(full_mass)
    return full_mass



# if __name__ == '__main__':
#     main()