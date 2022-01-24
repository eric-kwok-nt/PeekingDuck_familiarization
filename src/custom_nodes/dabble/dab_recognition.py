from typing import Any, Dict, List, Tuple

from peekingduck.pipeline.nodes.node import AbstractNode

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pdb


class Node(AbstractNode):
    """This is a custom node of dab detection

    Args:
        config (:obj:`Dict[str, Any]` | :obj:`None`): Node configuration.
    """

    def __init__(self, config: Dict[str, Any] = None, **kwargs: Any) -> None:
        super().__init__(config, node_path=__name__, **kwargs)
        self.nose = np.array([-1.,-1.])
        self.left_eye = np.array([-1.,-1.])
        self.right_eye = np.array([-1.,-1.])
        self.lower_left_arm_vec = np.array([-1.,-1.])
        self.lower_right_arm_vec = np.array([-1.,-1.])
        self._normalize_scores()

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:  # type: ignore
        """This node does dab detection.

        Args:
            inputs (dict): Dictionary with keys "keypoints".

        Returns:
            outputs (dict): Dictionary with keys "obj_tags".
        """
        keypoints = inputs['keypoints']
        is_dab_list = ['Waiting...' for _ in keypoints]
        for i, keypoint in enumerate(keypoints):
            left_wrist = keypoint[self.keypoints_id['left_wrist']]
            right_wrist = keypoint[self.keypoints_id['right_wrist']]
            left_elbow = keypoint[self.keypoints_id['left_elbow']]
            right_elbow = keypoint[self.keypoints_id['right_elbow']]
            left_shoulder = keypoint[self.keypoints_id['left_shoulder']]
            right_shoulder = keypoint[self.keypoints_id['right_shoulder']]
            self.nose = keypoint[self.keypoints_id['nose']]
            self.left_eye = keypoint[self.keypoints_id['left_eye']]
            self.right_eye = keypoint[self.keypoints_id['right_eye']]
            
            if not (self._are_keypoints_present(
                left_wrist, right_wrist, left_elbow, right_elbow, left_shoulder, right_shoulder
            )):
                
                self.lower_left_arm_vec = left_wrist - left_elbow
                self.lower_right_arm_vec = right_wrist - right_elbow
            
                is_left_straight = self._is_straight_arm(self.lower_left_arm_vec, left_elbow, left_shoulder)
                is_right_straight = self._is_straight_arm(self.lower_right_arm_vec, right_elbow, right_shoulder)
                is_left_bent = self._is_bent_arm(self.lower_left_arm_vec, left_elbow, left_shoulder)
                is_right_bent = self._is_bent_arm(self.lower_right_arm_vec, right_elbow, right_shoulder)
                is_head_close_left_wrist = self._is_head_close_to_wrist_or_elbow(left_wrist, left_elbow)
                is_head_close_right_wrist = self._is_head_close_to_wrist_or_elbow(right_wrist, right_elbow)
                are_arms_parallel = self._are_lowerarms_parallel()

                if are_arms_parallel:
                    if is_left_straight and is_right_bent and is_head_close_right_wrist:
                        total_score = is_left_straight*self.score_weightage['straight_arm'] +\
                            is_right_bent*self.score_weightage['bent_arm'] +\
                            is_head_close_right_wrist*self.score_weightage['head_wrist'] +\
                            are_arms_parallel*self.score_weightage['lower_arm_parallel']
                        
                        is_dab_list[i] = f"Dab Detected! Score: {total_score:.2f}"
                    
                    elif is_right_straight and is_left_bent and is_head_close_left_wrist:
                        total_score = is_right_straight*self.score_weightage['straight_arm'] +\
                            is_left_bent*self.score_weightage['bent_arm'] +\
                            is_head_close_left_wrist*self.score_weightage['head_wrist'] +\
                            are_arms_parallel*self.score_weightage['lower_arm_parallel']
                        
                        is_dab_list[i] = f"Dab Detected! Score: {total_score:.2f}"

        outputs = {
            "obj_tags": is_dab_list
        }

        return outputs

    def _is_straight_arm(
        self, lower_arm_vec: np.ndarray, elbow: List[float], shoulder: List[float]
        ) -> Tuple[float, bool]:
        """This method checks if the arm is straight

        Args:
            lower_arm_vec (np.ndarray): Vector of lower arm from elbow to wrist [float, float]
            elbow (List[float]): Coordinates of the elbow position [1x2]
            shoulder (List[float]): Coordinates of the shoulder position [1x2]

        Returns:
            Tuple[float, bool]: Outputs the score of how straight the arm is, and False if it is below threshold
        """
        upper_arm_vec = [elbow - shoulder]
        straight_score = cosine_similarity([lower_arm_vec], upper_arm_vec).ravel()[0]
        if straight_score >= self.thresholds['straight_arm']:
            return straight_score
        else:
            return False
    
    def _is_bent_arm(
        self, lower_arm_vec: np.ndarray, elbow: List[float], shoulder: List[float]
    ) -> Tuple[float, bool]:
        """This method checks if the arm is bent

        Args:
            lower_arm_vec (np.ndarray): Vector of lower arm from elbow to wrist [float, float]
            elbow (List[float]): Coordinates of the elbow position [1x2]
            shoulder (List[float]): Coordinates of the shoulder position [1x2]

        Returns:
            Tuple[float, bool]: Outputs the score of the extend of arm bending, and False if it is below threshold
        """
        upper_arm_vec = [shoulder - elbow]
        bent_score = cosine_similarity([lower_arm_vec], upper_arm_vec).ravel()[0]
        if bent_score >= self.thresholds['bent_arm']:
            return bent_score
        else:
            return False

    def _is_head_close_to_wrist_or_elbow(
        self, wrist: List[float], elbow: List[float]
    ) -> float:
        """This method checks if the eyes or nose is close to the wrist or elbow

        Args:
            wrist (List[float]): Coordinates of the wrist position [1x2]
            elbow (List[float]): Coordinates of the elbow position [1x2]

        Returns:
            float: Outputs the score of how close the eyes or nose is to the wrist or arm, and False if it is below threshold
        """
        lower_arm_len = np.linalg.norm(wrist-elbow)
        items_to_check = [self.nose, self.left_eye, self.right_eye]
        head_arm_score = 0
        for item in items_to_check:
            if not (item == [-1., -1.]).all():
                item_wrist_dist = np.linalg.norm(item-wrist)
                item_elbow_dist = np.linalg.norm(item-elbow)
                dist = min([item_wrist_dist/lower_arm_len, item_elbow_dist/lower_arm_len])
                if dist <= self.thresholds['head_wrist'] and 1-dist > head_arm_score:
                    head_arm_score = 1-dist
        
        return head_arm_score

    def _are_lowerarms_parallel(self) -> Tuple[float, bool]:
        """This method checks if the lower arms are parallel

        Returns:
            Tuple[float, bool]: Outputs the score of how parallel the lower arms to each other, and False if it is below threshold
        """
        parallel_score = cosine_similarity([self.lower_left_arm_vec], [self.lower_right_arm_vec]).ravel()[0]
        if parallel_score >= self.thresholds['lower_arm_parallel']:
            return parallel_score
        return False

    def _are_keypoints_present(
        self, left_wrist,right_wrist, left_elbow, right_elbow, left_shoulder, right_shoulder
        ) -> bool:
        """Checks whether the keypoints are detected

        Args:
            left_wrist (np.ndarray): Coordinate of left_wrist [1x2]
            right_wrist (np.ndarray): Coordinate of right_wrist [1x2]
            left_elbow (np.ndarray): Coordinate of left_elbow [1x2]
            right_elbow (np.ndarray): Coordinate of right_elbow [1x2]
            left_shoulder (np.ndarray): Coordinate of left_shoulder [1x2]
            right_shoulder (np.ndarray): Coordinate of right_shoulder [1x2]

        Returns:
            bool: Outputs True if left_wrist, right_wrist, left_elbow, right_elbow, left_shoulder, right_shoulder are all present, 
                and either nose or left_eye or right_eye are present
        """
        return (
            (left_wrist == [-1.,-1.]).all() or
            (right_wrist == [-1.,-1.]).all() or
            (left_elbow == [-1.,-1.]).all() or
            (right_elbow == [-1.,-1.]).all() or
            (left_shoulder == [-1.,-1.]).all() or
            (right_shoulder == [-1.,-1.]).all() or
            (
                (self.nose == [-1.,-1.]).all() and 
                (self.left_eye == [-1.,-1.]).all() and
                (self.right_eye == [-1.,-1.]).all()
                )
        )
    
    def _normalize_scores(self):
        """Normalizes the score weightage to sum of 1
        """
        total = 0
        for k, v in self.score_weightage.items():
            total += v
        for k, v in self.score_weightage.items():
            self.score_weightage[k] /= total
