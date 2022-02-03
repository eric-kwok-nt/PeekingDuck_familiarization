# PeekingDuck_familiarization
## 1. Description

This mini project attempts to test out the Human-Pose Estimation and Object Detection features in [PeekingDuck](https://github.com/aimakerspace/PeekingDuck) library

### 1.1 Dab Detection example using HRNet + YOLOv4
![An example for dab detection](./images/dab_detection.gif)

## 2. Project Organization
------------
    .
    ├── LICENSE
    ├── README.md
    ├── config                   -> Contains yaml config files for various experiments
    ├── dab_detection_demo.ipynb -> Notebook for demo
    ├── environment.yml
    ├── images                   -> images for README file
    └── src                      -> Contains yaml config files and python scripts for custom nodes
        ├── __init__.py
        ├── custom_nodes         
        │   ├── configs          
        └── data                 -> Contains yaml config files for recording videos with dab detection
--------

## 3. Usage
### 3.1 Testing out Dab Detection using PeekingDuck CLI 
#### 3.1.1 Dab Detection with Live Input
Using __HRNet + YOLOv4__:
```bash
peekingduck run --config_path ./config/dab_live_hrnet_config.yml
```
Using __PoseNet__:
```bash
peekingduck run --config_path ./config/dab_live_posenet_config.yml
```

#### 3.1.2 Dab Detection with Recorded video
Command to run dab detection using __HRNet + YOLOv4__:
```bash
peekingduck run --config_path ./config/dab_recorded_hrnet_config.yml
```
Command to run dab detection using __PoseNet__:
```bash
peekingduck run --config_path ./config/dab_recorded_posenet_config.yml
```
By default, it will look for the video file in the following path ```./data/raw/video1.mp4```. It can be changed manually in the respective yaml configuration file as such:
```yaml
nodes:
- input.recorded:
    input_dir: "path_to_your_video_file"
```
The output video will be in ```data/processed/``` by default. Again, it can be changed manually in the respective yaml configuration file as such:
```yaml
nodes:
- output.media_writer:
    output_dir: "path_to_your_video_file"
```

## 4. Method
### 4.1 Dab Detection
The Dab Detection process is based on some of the Pose Estimation / Object Detection models (YOLOv4 + HRNet or PoseNet) that comes with the [PeekingDuck](https://github.com/aimakerspace/PeekingDuck) library. 

For the full list of models available and their coresponding performances, please refer to the links [Object Detection Models](https://peekingduck.readthedocs.io/en/stable/resources/01a_object_detection.html#general-object-detection-ids) and [Pose Estimation Models](https://peekingduck.readthedocs.io/en/stable/resources/01b_pose_estimation.html#whole-body-keypoint-ids).

By using the Pose Estimation Model and obtaining the respective pose coordinates, a [custom](https://peekingduck.readthedocs.io/en/stable/getting_started/03_custom_nodes.html#) "Dabble" node was created with a set of rules to determine if the detected pose coordinates are indeed from a Dab Move as shown below:
<br><img src="./images/dabbing.png" alt="A Dabbing dance move" width=180>
_Image by <a href="https://pixabay.com/users/mohamed_hassan-5229782/?utm_source=link-attribution&amp;utm_medium=referral&amp;utm_campaign=image&amp;utm_content=5185246">mohamed Hassan</a> from <a href="https://pixabay.com/?utm_source=link-attribution&amp;utm_medium=referral&amp;utm_campaign=image&amp;utm_content=5185246">Pixabay</a>_ <br>

#### 4.1.1 Dab Detection Heuristics
1. One of the arms has to be bent while the other arm has to be straightened
2. The head (nose or left eye, or right eye) has to be close to either the elbow or the wrist of the bent arm.
3. Both lower arms have to be almost parallel

In addition, if any of the keypoints coordinates - wrists AND shoulders AND elbows AND (left eye or right eye or nose) is __missing__ from the pose estimation output, it will __not__ detect any dab action.

#### 4.1.2 Scoring
The scoring of each dab is based on how close the detected poses are to each of the rules above. For checking whether the arms are bent or straight and whether the lower arms are parallel, cosine similarity functions are used. 

For checking if the head is close to the wrist or elbow, L2-norm distance is used and it is normalized by the length of the lower arm. 

Each of the scores has to be above or below certain predefined threshold to be considered successful. The set of predefined thresholds can be changed in the configuration file ```src\custom_nodes\configs\dabble\dab_recognition.yml``` in the following:
```yaml
thresholds:
  straight_arm: 0.8       # [0, 1] More is better
  bent_arm: 0.4           # [0, 1] More is better
  head_wrist: 0.4         # [0, 1] Less is better. How close the head is to either the wrist or elbow
  lower_arm_parallel: 0.8 # [0, 1] More is better
```
The final score is the weighted sum of the score from the individual rules. The weightage of each score for the respective rules can be changed by the user in the yaml configuration file ```src\custom_nodes\configs\dabble\dab_recognition.yml``` in the following:
```yaml
score_weightage: 
  straight_arm: 0.25
  bent_arm: 0.25
  head_wrist: 0.25
  lower_arm_parallel: 0.25
```
The final score is being calculated from the formula: (1 - head_wrist_threshold) $\times$ head_wrist_weightage + $\Sigma$ (remaining_thresholds $\times$ score_weigtage)

It is to note that the lower bound of the score is dependent on the threshold being set _(i.e. lower threshold will result in a reduced lower bound of the score)_.

In addition, the sum of the score weightage do not need to be 1. A function has been included in the process to automatically normalize the total score weightage to 1.

### 4.2 Bus Passenger Counting
- [ ] To be updated  

## 5. Author and Acknowledgements
**Author**
* [Eric Kwok](https://github.com/eric-kwok-nt)

This mini-project is created using the [PeekingDuck](https://github.com/aimakerspace/PeekingDuck) library.


