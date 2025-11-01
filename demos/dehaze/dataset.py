
# dataset
import os
import random

def load_dataset(hazy_dir, gt_dir):
    hazy_files = sorted([f for f in os.listdir(hazy_dir) if not f.startswith('.')])
    gt_files   = sorted([f for f in os.listdir(gt_dir) if not f.startswith('.')])
    samples = []
    for h, g in zip(hazy_files, gt_files):
        sample = f"{os.path.join(hazy_dir, h)} {os.path.join(gt_dir, g)}"
        samples.append(sample)
    return samples

def split_dataset(samples, val_ratio=0.1, test_ratio=0.1):
    random.shuffle(samples)
    N = len(samples)
    val_num = int(N * val_ratio)
    test_num = int(N * test_ratio)
    train_num = N - val_num - test_num
    train = samples[:train_num]
    val = samples[train_num:train_num + val_num]
    test = samples[train_num + val_num:]
    return train, val, test

def write_txt(samples, save_path):
    with open(save_path, 'w') as f:
        for line in samples:
            f.write(line + '\n')

if __name__ == '__main__':
    hazy_dir = './test-images/hazy'
    gt_dir   = './test-images/gt'
    out_dir  = './test-images/split_txt'
    os.makedirs(out_dir, exist_ok=True)
    samples = load_dataset(hazy_dir, gt_dir)
    train, val, test = split_dataset(samples, 0.2, 0.2)
    print(f'train: {len(train)}, val: {len(val)}, test: {len(test)}')
    write_txt(train, os.path.join(out_dir, 'train.txt'))
    write_txt(val,   os.path.join(out_dir, 'val.txt'))
    write_txt(test,  os.path.join(out_dir, 'test.txt'))
    print("dataset file generated！")
