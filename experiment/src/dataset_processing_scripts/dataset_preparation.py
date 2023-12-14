import os

import pandas as pd
import config

def import_cbis_dataset_mass():
    csv_root = '/content/drive/MyDrive/archive/csv'
    img_root = '/content/drive/MyDrive/archive/jpeg'
    # csv_root = '/Users/aikansh_yadav/Documents/Portable-Skin-Lesion-Diagnosis-main/archive/csv'
    # img_root = '/Users/aikansh_yadav/Documents/Portable-Skin-Lesion-Diagnosis-main/archive/jpeg'
    
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

    # print(f'Shape of mass_train: {mass_train.shape}')
    # print(f'Shape of mass_test: {mass_test.shape}')

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

def import_cbis_dataset_calc():
    csv_root = '/content/drive/MyDrive/archive/csv'
    img_root = '/content/drive/MyDrive/archive/jpeg'
    # csv_root = '/Users/aikansh_yadav/Documents/Portable-Skin-Lesion-Diagnosis-main/archive/csv'
    # img_root = '/Users/aikansh_yadav/Documents/Portable-Skin-Lesion-Diagnosis-main/archive/jpeg'

    df_dicom = pd.read_csv(csv_root + '/' + 'dicom_info.csv')

    if df_dicom.empty:
        print('DataFrame is empty!')

    cropped_images=df_dicom[df_dicom.SeriesDescription == 'cropped images'].image_path
    cropped_images = cropped_images.apply(lambda x: x.replace('CBIS-DDSM/jpeg', img_root))

    full_mammogram_images=df_dicom[df_dicom.SeriesDescription == 'full mammogram images'].image_path
    full_mammogram_images = full_mammogram_images.apply(lambda x: x.replace('CBIS-DDSM/jpeg', img_root))

    ROI_mask_images=df_dicom[df_dicom.SeriesDescription == 'ROI mask images'].image_path
    ROI_mask_images = ROI_mask_images.apply(lambda x: x.replace('CBIS-DDSM/jpeg', img_root))

    # print(cropped_images.iloc[0])

    # print(cropped_images.head())
    # print(full_mammogram_images.head())
    # print(ROI_mask_images.head())

    # organize image paths
    full_mammo_dict = dict()
    cropped_images_dict = dict()
    roi_img_dict = dict()

    for dicom in full_mammogram_images:
        key = dicom.split("/")[6]
        full_mammo_dict[key] = dicom
    for dicom in cropped_images:
        key = dicom.split("/")[6]
        cropped_images_dict[key] = dicom
    for dicom in ROI_mask_images:
        key = dicom.split("/")[6]
        roi_img_dict[key] = dicom

    calc_train = pd.read_csv(csv_root + '/' + 'calc_case_description_train_set.csv')
    calc_test = pd.read_csv(csv_root + '/' + 'calc_case_description_train_set.csv')

    def fix_image_path(data):
        for index, img in enumerate(data.values):
            img_name = img[11].split("/")[2]
            if(full_mammo_dict.get(img_name) is not None):
                data.iloc[index,11] = full_mammo_dict[img_name]
            
            img_name = img[12].split("/")[2]
            if(cropped_images_dict.get(img_name) is not None):
                data.iloc[index,12] = cropped_images_dict[img_name]
        
    # apply to datasets
    fix_image_path(calc_train)
    fix_image_path(calc_test)

    # print(calc_train.info())
    # print(calc_test.info())

    # check unique values in pathology column
    # print(calc_train.pathology.unique())

    # print(calc_train.info())

    calc_train = calc_train.rename(columns={'left or right breast': 'left_or_right_breast',
                                            'breast density' : 'breast_density',
                                           'image view': 'image_view',
                                           'abnormality id': 'abnormality_id',
                                           'abnormality type': 'abnormality_type',
                                           'calc type': 'calc_type',
                                           'calc distribution': 'calc_distribution',
                                           'image file path': 'image_file_path',
                                           'cropped image file path': 'cropped_image_file_path',
                                           'ROI mask file path': 'ROI_mask_file_path'})

    # print(calc_train.head(5))

    # print(calc_train.isnull().sum())

    # fill in missing values using the backwards fill method
    calc_train['calc_type'] = calc_train['calc_type'].fillna(method='bfill')
    calc_train['calc_distribution'] = calc_train['calc_distribution'].fillna(method='bfill')

    #check null values
    # print(calc_train.isnull().sum())

    # check datasets shape
    # print(f'Shape of calc_train: {calc_train.shape}')
    # print(f'Shape of calc_test: {calc_test.shape}')

    calc_test = calc_test.rename(columns={'left or right breast': 'left_or_right_breast',
                                            'breast density' : 'breast_density',
                                           'image view': 'image_view',
                                           'abnormality id': 'abnormality_id',
                                           'abnormality type': 'abnormality_type',
                                           'calc type': 'calc_type',
                                           'calc distribution': 'calc_distribution',
                                           'image file path': 'image_file_path',
                                           'cropped image file path': 'cropped_image_file_path',
                                           'ROI mask file path': 'ROI_mask_file_path'})
    
    # print(calc_test.isnull().sum())

    # fill in missing values using the backwards fill method
    calc_test['calc_type'] = calc_test['calc_type'].fillna(method='bfill')
    calc_test['calc_distribution'] = calc_test['calc_distribution'].fillna(method='bfill')

    # print(calc_test.isnull().sum())

    full_calc = pd.concat([calc_train, calc_test], axis=0)
    print("Returning data from dataset_preparation method")
    return full_calc

def import_cbis_dataset():
    if(config.mammogram_type == "Mass"):
        print("Mammography type : Mass")
        return import_cbis_dataset_mass()
    elif(config.mammogram_type == "Calc"):
        print("Mammography type : Calc")
        return import_cbis_dataset_calc()
    else:
        mass = import_cbis_dataset_mass()
        calc = import_cbis_dataset_calc()
        print("Mammography type : All")
        all = pd.concat([mass, calc], axis=0)
        return all

# if __name__ == '__main__':
#     import_cbis_dataset()