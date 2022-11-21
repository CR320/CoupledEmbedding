# Regularizing Vector Embedding in Bottom-Up Human Pose Estimation

> [**Regularizing Vector Embedding in Bottom-Up Human Pose Estimation**],            
> Haixin Wang, Lu Zhou, Yingying Chen, Ming Tang, jinqiao Wang  
> In: European Conference on Computer Vision (ECCV), 2022   

## Introduction
This is the official code of [Regularizing Vector Embedding in Bottom-Up Human Pose Estimation](https://link.springer.com/chapter/10.1007/978-3-031-20068-7_7).  
In this paper, we probe the limitation of keypoint grouping in Associative Embedding \cite{newell2017associative} and find that different dimensions of embeddings  in an image are highly linearly correlated. To address this issue, we propose a novel keypoint grouping method named Coupled Embedding for bottom-up human pose estimation. Our method imposes an additional constraint on embeddings to learn sparse multidimensional embeddings and improve heatmap regression with the embedded scale information.  Our method achieves outstanding performance on CrowdPose Test and COCO Test-dev. We conduct a series of experiments and the results show that our model has significant advantages in complex scenes.

## Main Results
### Results on COCO test-dev2017
| Model               | Input size | Multi-Scale Test |  AP  | AP .5 | AP .75 | AP (M) | AP (L) |
|---------------------|------------|------------------|------|-------|--------|--------|--------|
| **ce_hrnet_w32**    |  512       |&#10008;          | 67.0 | 88.9  | 73.7   | 60.4   | 76.4   |
| **ce_hrnet_w48**    |  640       |&#10008;          | 68.4 | 88.7  | 75.5   | 63.8   | 75.9   |
| **ce_hrhrnet_w32**  |  512       |&#10008;          | 68.8 | 90.3  | 75.2   | 62.9   | 77.1   |
| **ce_hrhrnet_w48**  |  640       |&#10008;          | 71.1 | 90.8  | 77.8   | 66.4   | 78.0   |
| **ce_hrhrnet_w48**  |  640       |&#10004;          | 72.8 | 91.2  | 79.9   | 68.3   | 79.3   |

### Results on CrowdPose test
| Method               | Input size | Multi-Scale Test |  AP  | AP .5 | AP .75 | AP (E) | AP (M) | AP (H) |
|----------------------|------------|------------------|------|-------|--------|--------|--------|--------|
| **ce_hrnet_w32**     | 512        |&#10008;          | 68.9 | 89.0  | 74.2   | 76.3   | 69.5   | 60.8   |
| **ce_hrnet_w48**     | 640        |&#10008;          | 70.1 | 89.8  | 75.5   | 77.5   | 70.8   | 62.2   |
| **ce_hrhrnet_w32**   | 512        |&#10008;          | 69.6 | 89.7  | 74.9   | 76.9   | 70.3   | 61.6   |
| **ce_hrhrnet_w48**   | 640        |&#10008;          | 70.5 | 89.9  | 76.0   | 77.7   | 71.1   | 62.4   |
| **ce_hrhrnet_w48**   | 640        |&#10004;          | 71.6 | 90.1  | 77.3   | 79.0   | 72.2   | 63.3   |
| **ce_hrhrnet_w48+**  | 640        |&#10008;          | 72.9 | 89.5  | 78.8   | 79.6   | 73.7   | 64.5   |
| **ce_hrhrnet_w48+**  | 640        |&#10004;          | 74.5 | 91.1  | 80.2   | 81.3   | 75.4   | 66.2   |

*Note: Flipping test is used. + indicates the model is pretrained on COCO.*

## Model Zoo
Please download models from [Google Drive](https://drive.google.com/drive/folders/1Jln6GtSoFIxbwt6hQ3YLXt_-a0dgIp0P) or [Baidu Netdisk](https://pan.baidu.com/s/1zBbhPQTwW0JxZl1qq7QNlA) (password: 7amz)

### Data preparation
**For COCO data**, please download from [COCO](http://cocodataset.org/#download), 2017 Train/Val is needed for COCO keypoints training and validation. 
**For CrowdPose data**, please download from [CrowdPose](https://github.com/Jeff-sjtu/CrowdPose#dataset), Train/Val is needed for CrowdPose keypoints training.
Extract them under ${YOUR_DATA_DIR}, and make them look like this:
```
${YOUR_DATA_DIR}
    |-- coco
    `-- |-- annotations
        |   |-- person_keypoints_train2017.json
        |   |-- person_keypoints_val2017.json
        |   `-- image_info_test-dev2017.json
        |-- train2017
        |-- val2017
        `-- test2017
            
    |-- crowdpose
    `-- |-- json
        |   |-- crowdpose_train.json
        |   |-- crowdpose_val.json
        |   |-- crowdpose_trainval.json (generated by tools/crowdpose_concat_train_val.py)
        |   `-- crowdpose_test.json
        `-- images
```

## Quick start
1. Clone this repo, and your directory tree should look like this:

   ```
   CoupledEmbedding
   ├── config
   ├── datasets
   ├── models
   ├── pretrained
   ├── README.md
   ├── requirements.txt
   ├── test.py
   ├── train.txt
   └── utils.py
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Download pretrained backbone from openmmlab: 
   ```
   wget https://download.openmmlab.com/mmpose/pretrain_models/hrnet_w32-36af842e.pth
   wget https://download.openmmlab.com/mmpose/pretrain_models/hrnet_w48-8ef0771d.pth
   ```
   The directory should look like this:
   ```
   CoupledEmbedding
   |-- pretrained
   `-- |-- hrnet_w32-36af842e.pth
       `-- hrnet_w48-8ef0771d.pth
   ```
4. Training with multiple gpus (take COCO as the example):

```
python train.py \
    config/coco/hrnet32_512.py --num_gpus ${NUM_GPUS} --distributed True --work_dir ${YOUR WORK DIR}
```
5. Testing with single gpu on validation set:

```
python test.py \
    config/coco/hrnet32_512.py ${YOUR WORK DIR}/coco/ce_hrnet_w32.pth --set val
```
6. Testing with single gpu on test set:

```
python test.py \
    config/coco/hrnet32_512.py ${YOUR WORK DIR}/coco/ce_hrnet_w32.pth --set test
```
7. Testing with multiple gpus:

```
python test.py \
    config/coco/hrnet32_512.py ${YOUR WORK DIR}/coco/ce_hrnet_w32.pth --set test --num_gpus ${NUM_GPUS} --distributed True
```
