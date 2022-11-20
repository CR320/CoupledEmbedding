# Regularizing Vector Embedding in Bottom-Up Human Pose Estimation

> [**Regularizing Vector Embedding in Bottom-Up Human Pose Estimation**],            
> Haixin Wang, Lu Zhou, Yingying Chen, Ming Tang, jinqiao Wang  
> In: European Conference on Computer Vision (ECCV), 2022   

## Main Results
### Results on COCO test-dev2017
| Model | Input size | AP | AP .5 | AP .75 | AP (M) | AP (L) |
|--------------------|------------|---------|--------|-------|-------|--------|
| **pose_hrnet_w32** |  512 | 67.0 | 88.9 | 73.7 | 60.4 | 76.4 |
| **pose_hrnet_w48** |  640 | 68.4 | 88.7 | 75.5 | 63.8 | 75.9 |
| **pose_hrhrnet_w32** |  512 | 68.8 | 90.3 | 75.2 | 62.9 | 77.1 |
| **pose_hrhrnet_w48** |  640 | 71.1 | 90.8 | 77.8 | 66.4 | 78.0 |
| **pose_hrhrnet_w48\*** |  640 | 72.8 | 91.2 | 79.9 | 68.3 | 79.3 |
*Note: Superscript \∗ means multiscale test. Flipping test is used.*
