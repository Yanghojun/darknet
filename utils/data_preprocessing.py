import numpy as np
from sklearn.model_selection import train_test_split
from glob import glob
import os
from shutil import copyfile

def copy_files_to_dir(file_paths:list, dir_path:str):
    for path in file_paths:
        file_name = path.split('/')[-1]
        copyfile(path, dir_path + f'/{file_name}')

def check_files_existence(paths:list):
    for path in paths:
        if not os.path.exists(path):
            print(f'{path} is not exist', end=' ')
            return False
    return True

def data_split(dir_path:str, save_train_path:str, save_valid_path:str):
    """
        1. Process train_test_split data in dir_path
        2. Make directory based on save_train_path, save_valid_path
        3. copy splited data to each directory

    Args:
        dir_path (str): _description_
        save_train_path (str): _description_
        save_valid_path (str): _description_
    """
    img_paths = glob(dir_path+'/*.png')
    
    # use img datas to read txt files that have same name
    txt_paths = []
    for path in img_paths:
        txt_paths.append(path[:-3] + 'txt')
        
    assert check_files_existence(txt_paths), 'Error!'
    
    train_img_paths, val_img_paths, train_txt_paths, val_txt_paths = train_test_split(img_paths, txt_paths, random_state=5, test_size=0.1)
    os.makedirs(save_train_path, exist_ok=True), os.makedirs(save_valid_path, exist_ok=True)
    copy_files_to_dir(train_img_paths, save_train_path), copy_files_to_dir(train_txt_paths, save_train_path)
    copy_files_to_dir(val_img_paths, save_valid_path), copy_files_to_dir(val_txt_paths, save_valid_path)
    
def make_train_txt_file(train_path, valid_path, save_path):
    paths = glob(f'{train_path}/*.png')
    with open(os.path.join(save_path, 'train.txt'), 'w') as f:
        for path in paths:
            path = path.split('/')
            path = os.path.join(train_path, path[-1])
            f.write(path + '\n')
            
    paths = glob(f'{valid_path}/*.png')
    with open(os.path.join(save_path, 'valid.txt'), 'w') as f:
        for path in paths:
            path = path.split('/')
            path = os.path.join(valid_path, path[-1])
            f.write(path + '\n')
            
            
def las_preprocess(data_dir):
    # record image paths
    paths = glob(f'./{data_dir}/*.png')

    # preprocessing (use only third channel)
    for path in tqdm(paths, desc='Creating preprocessed png files...', ascii=True):
        # print(path, img.shape)
        img = iio.v2.imread(path)
        img = img[:,:,2]
        img = np.expand_dims(img, axis=2)
        img = np.concatenate((img, img, img), axis=2)
        iio.v2.imwrite(path, img)

if __name__ == '__main__':
    # train_test_split
    # data_split('./data/las_data_23_02_09_300_data_annotation', './data/train_las', './data/valid_las')
    
    # create train_txt file
    make_train_txt_file('./data/train_las', './data/valid_las', './data')