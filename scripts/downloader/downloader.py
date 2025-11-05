import os
import shutil

import zipfile
import requests

from tqdm import tqdm

import argparse


def get_files(debug=False):
    train_files = '''
        adapter02_indoor
        bag03_indoor
        bag05_indoor
        ball02_indoor
        ball03_indoor
        ball04_indoor
        ball05_indoor
        ball07_indoor
        ball08_wild
        ball09_wild
        ball12_wild
        ball13_indoor
        ball14_wild
        ball17_wild
        ball19_indoor
        ball21_indoor
        basket_indoor
        beautifullight01_indoor
        bike01_wild
        bike02_wild
        bike03_wild
        book01_indoor
        book02_indoor
        book04_indoor
        book05_indoor
        book06_indoor
        bottle01_indoor
        bottle02_indoor
        bottle05_indoor
        bottle06_indoor
        box_indoor
        candlecup_indoor
        car01_indoor
        car02_indoor
        cart_indoor
        cat02_indoor
        cat03_indoor
        cat04_indoor
        cat05_indoor
        chair01_indoor
        chair02_indoor
        clothes_indoor
        colacan01_indoor
        colacan02_indoor
        colacan04_indoor
        container01_indoor
        container02_indoor
        cube01_indoor
        cube04_indoor
        cube06_indoor
        cup03_indoor
        cup05_indoor
        cup06_indoor
        cup07_indoor
        cup08_indoor
        cup09_indoor
        cup10_indoor
        cup11_indoor
        cup13_indoor
        cup14_indoor
        duck01_wild
        duck02_wild
        duck04_wild
        duck05_wild
        duck06_wild
        dumbbells02_indoor
        earphone02_indoor
        egg_indoor
        file02_indoor
        flower01_indoor
        flower02_wild
        flowerbasket_indoor
        ghostmask_indoor
        glass02_indoor
        glass03_indoor
        glass04_indoor
        glass05_indoor
        guitarbag_indoor
        gymring_wild
        hand02_indoor
        hat01_indoor
        hat02_indoor_320
        hat03_indoor
        hat04_indoor
        human01_indoor
        human03_wild
        human04_wild
        human05_wild
        human06_indoor
        leaves01_wild
        leaves02_indoor
        leaves03_wild
        leaves04_indoor
        leaves05_indoor
        leaves06_wild
        lock01_wild
        mac_indoor
        milkbottle_indoor
        mirror_indoor
        mobilephone01_indoor
        mobilephone02_indoor
        mobilephone04_indoor
        mobilephone05_indoor
        mobilephone06_indoor
        mushroom01_indoor
        mushroom02_wild
        mushroom03_wild
        mushroom04_indoor
        mushroom05_indoor
        notebook02_indoor
        notebook03_indoor
        paintbottle_indoor
        painting_indoor_320
        parkingsign_wild
        pigeon03_wild
        pigeon06_wild
        pigeon07_wild
        pine01_indoor
        pine02_wild_320
        shoes01_indoor
        shoes03_indoor
        skateboard01_indoor
        skateboard02_indoor
        speaker_indoor
        stand_indoor
        suitcase_indoor
        swing01_wild
        swing02_wild
        teacup_indoor
        thermos01_indoor
        thermos02_indoor
        toiletpaper02_indoor
        toiletpaper03_indoor
        toiletpaper04_indoor
        toy01_indoor
        toy04_indoor
        toy05_indoor
        toy06_indoor
        toy07_indoor_320
        toy08_indoor
        toy10_indoor
        toydog_indoor
        trashbin_indoor
        tree_wild
        trophy_indoor
        ukulele02_indoor
        '''.strip().split()

    val_files = '''
        toy03_indoor
        pigeon05_wild
        bottle03_indoor
        ball16_indoor
        bag04_indoor
        '''.strip().split()

    test_files = '''
        adapter01_indoor
        backpack_indoor
        bag01_indoor
        bag02_indoor
        ball01_wild
        ball06_indoor
        ball10_wild
        ball11_wild
        ball15_wild
        ball18_indoor
        ball20_indoor
        bandlight_indoor
        beautifullight02_indoor
        book03_indoor
        bottle04_indoor
        card_indoor
        cat01_indoor
        colacan03_indoor
        cube02_indoor
        cube03_indoor
        cube05_indoor
        cup01_indoor
        cup02_indoor
        cup04_indoor
        cup12_indoor
        developmentboard_indoor
        duck03_wild
        dumbbells01_indoor
        earphone01_indoor
        file01_indoor
        flag_indoor
        glass01_indoor
        hand01_indoor
        human02_indoor
        lock02_indoor
        mobilephone03_indoor
        notebook01_indoor
        pigeon01_wild
        pigeon02_wild
        pigeon04_wild
        pot_indoor
        roller_indoor
        shoes02_indoor
        squirrel_wild
        stick_indoor
        toiletpaper01_indoor
        toy02_indoor
        toy09_indoor
        ukulele01_indoor
        yogurt_indoor
        '''.strip().split()

    if debug:
        train_files = '''
            adapter02_indoor
            '''.strip().split()

        val_files = '''
            toy03_indoor        
            '''.strip().split()

        test_files = '''
            adapter01_indoor
            '''.strip().split()
    
    return train_files, val_files, test_files


def remove_invalid_files(files, split_dir):
    valid_files = [*files]
    invalid_files = []

    for f in os.listdir(split_dir):
        file_path = os.path.join(split_dir, f)

        # remove files (zip files)
        if os.path.isfile(file_path):
            s = input(f'WARNING: deleting old file {file_path}, enter "y" to continue: ')
            assert s == 'y'
            os.remove(file_path)
            continue
        
        # remove incomplete directories
        seq_path = os.path.join(split_dir, f)
        color_path = os.path.join(split_dir, f, 'color')
        depth_path = os.path.join(split_dir, f, 'depth')
        color_count = len(os.listdir(color_path))
        depth_count = len(os.listdir(depth_path))
        try:
            assert color_count == depth_count
            assert color_count > 0
        except:
            s = input(f'WARNING: deleting incomplete directory {seq_path} ({color_count}, {depth_count}), enter "y" to continue: ')
            assert s == 'y'
            shutil.rmtree(seq_path)
            continue

        if f in files:
            invalid_files.append(f)
            valid_files.remove(f)

    return valid_files, invalid_files


def download_individual_file_unzip(dst: str, url: str):
    assert os.path.isdir(dst)
    
    zip_filename = os.path.join(dst, os.path.basename(url.split('?')[0]))
    
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise error if download fails
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(zip_filename, 'wb') as f, tqdm(
        total=total_size, unit='B', unit_scale=True, desc='Downloading', ncols=80
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))
    
    print(f"  extracting {zip_filename} to {dst}...")
    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall(dst)
    
    print(f"  deleting {zip_filename}...")
    os.remove(zip_filename)


def download_individual_files(files, split_dir):
    for f in files:
        print('============================================================')
        seq_path = os.path.join(split_dir, f)
        
        # download
        print(f'downloading {f} to {split_dir}')
        for depthtrack_split, id in [('train 01', 5794115), 
                                     ('train 02', 5837926), 
                                     ('test', 5792146)]:
            try:
                url = f'https://zenodo.org/records/{id}/files/' + f + '.zip?download=1' #01
                dst = split_dir
                download_individual_file_unzip(dst, url)
                print(f'successfully downloaded {seq_path} from {depthtrack_split}')
                break
            except:
                print(f'{f} not found in {depthtrack_split}')
        else:
            print(f'no valid urls found for {f}!')
            assert False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and manage dataset directory.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default='mnt/MVT/data',
        help="Path to the dataset directory"
    )

    args = parser.parse_args()
    data_dir = args.data_dir
    print('data_dir: ', data_dir)
    print()
    debug = False

    # check directory exists
    assert os.path.isdir(data_dir)
    depthtrack_dir = os.path.join(data_dir, 'depthtrack/')
    train_dir = os.path.join(depthtrack_dir, 'train/')
    val_dir = os.path.join(depthtrack_dir, 'train/')
    test_dir = os.path.join(depthtrack_dir, 'test/')
    try:
        assert os.path.isdir(depthtrack_dir)
        assert os.path.isdir(train_dir)
        assert os.path.isdir(val_dir)
        assert os.path.isdir(test_dir)
    except:
        s = input(f'deleting old files in {data_dir}, enter "y" to continue: ')
        assert s == 'y'
        shutil.rmtree(depthtrack_dir, ignore_errors=True)
        os.makedirs(depthtrack_dir, exist_ok=False)
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

    # get filenames to download
    train_files, val_files, test_files = get_files(debug=debug)

    # remove invalid existing files
    train_files, skipped_train_files = remove_invalid_files(train_files, train_dir)
    val_files, skipped_val_files = remove_invalid_files(val_files, val_dir)
    test_files, skipped_test_files = remove_invalid_files(test_files, test_dir)

    print()
    print('train files to download: ', train_files)
    print('train files skipped: ', skipped_train_files)
    print()
    print('val files to download: ', val_files)
    print('val files skipped: ', skipped_val_files)
    print()
    print('test files to download: ', test_files)
    print('test files skipped: ', skipped_test_files)
    print()

    print('============================================================')
    print(f'Downloading to {data_dir}')
    input(f'Press enter to begin: ')
    # assert False

    download_individual_files(train_files, train_dir)
    download_individual_files(val_files, val_dir)
    download_individual_files(test_files, test_dir)

    print('\ndone.')