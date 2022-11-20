# Regularizing Vector Embedding in Bottom-Up Human Pose Estimation

> [**Regularizing Vector Embedding in Bottom-Up Human Pose Estimation**],            
> Haixin Wang, Lu Zhou, Yingying Chen, Ming Tang, jinqiao Wang  
> In: European Conference on Computer Vision (ECCV), 2022   

## Main Results
### Results on COCO test-dev2017
| Model                 | Input size | AP  | AP .5 | AP .75 | AP (M) | AP (L) |
|-----------------------|------------|-----|-------|--------|--------|--------|
| **pose_hrnet_w32**    |  512       |67.0 | 88.9  | 73.7   | 60.4   | 76.4   |
| **pose_hrnet_w48**    |  640       |68.4 | 88.7  | 75.5   | 63.8   | 75.9   |
| **pose_hrhrnet_w32**  |  512       |68.8 | 90.3  | 75.2   | 62.9   | 77.1   |
| **pose_hrhrnet_w48**  |  640       |71.1 | 90.8  | 77.8   | 66.4   | 78.0   |
| **pose_hrhrnet_w48\***|  640       |72.8 | 91.2  | 79.9   | 68.3   | 79.3   |

*Note: Flipping test is used. Superscript ∗ indicates multiscale test.*

### Results on CrowdPose test
| Method                 | Input size | AP | Ap .5 | AP .75 | AP (E) | AP (M) | AP (H) |
|------------------------|------------|----|-------|--------|--------|--------|--------|
| **pose_hrnet_w32**     | 512        |68.9| 89.0  | 74.2   | 76.3   | 69.5   | 60.8   |
| **pose_hrnet_w32**     | 512        |70.1| 89.8  | 75.5   | 77.5   | 70.8   | 62.2   |
| **pose_hrhrnet_w32**   | 512        |69.6| 89.7  | 74.9   | 76.9   | 70.3   | 61.6   |
| **pose_hrhrnet_w48**   | 512        |70.5| 89.9  | 76.0   | 77.7   | 71.1   | 62.4   |
| **pose_hrhrnet_w48\*** | 512        |71.6| 90.1  | 77.3   | 79.0   | 72.2   | 63.3   |
| **pose_hrhrnet_w48+**  | 512        |72.9| 89.5  | 78.8   | 79.6   | 73.7   | 64.5   |
| **pose_hrhrnet_w48\*+**| 512        |74.5| 91.1  | 80.2   | 81.3   | 75.4   | 66.2   |

*Note: Flipping test is used. + indicates the model is pretrained on COCO. Superscript ∗ indicates multiscale test.*
