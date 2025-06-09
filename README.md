# PointKAN: Enhancing Point Cloud Analysis via Kolmogorov-Arnold Networks 

## News & Updates:

- [x] **06/09/2025**: codes release


## Install

```bash
# step 1. clone this repo
git clone https://github.com/huzhihen/pointKAN.git
cd pointKAN

# step 2. create a conda virtual environment and activate it
conda create -n pointkan python=3.8
conda activate pointkan
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch
pip install cycler einops h5py pyyaml==5.4.1 scikit-learn==0.24.2 scipy tqdm matplotlib==3.4.2
pip install pointnet2_ops_lib/.
```


## Useage

### Classification ModelNet40
**Train**:  The dataset will be automatically downloaded, run following command to train.

By default, it will create a folder named "checkpoints/{modelName}-{msg}-{randomseed}", which includes args.txt, best_checkpoint.pth, last_checkpoint.pth, log.txt, out.txt.
```bash
cd classification_ModelNet40
# train pointKAN
python main.py --model pointKAN
```


To conduct voting testing, run
```bash
# please modify the msg accrodingly
python voting.py --model pointKAN --msg demo
```


### Classification ScanObjectNN

**Train**:  The dataset will be automatically downloaded

By default, it will create a fold named "checkpoints/{modelName}-{msg}-{randomseed}", which includes args.txt, best_checkpoint.pth, last_checkpoint.pth, log.txt, out.txt.

```bash
cd classification_ScanObjectNN
# train pointKAN
python main.py --model pointKAN
```

### Part segmentation

- Make data folder and download the dataset
```bash
cd part_segmentation
mkdir data
cd data
wget https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip --no-check-certificate
unzip shapenetcore_partanno_segmentation_benchmark_v0_normal.zip
```

- Train pointKAN
```bash
# train pointKAN
python main.py --model pointKAN
```

+ Visualize

```bash
python visualize.py
```
