from typing import Any, Dict, List, Tuple

from peekingduck.pipeline.nodes.node import AbstractNode

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import cv2
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
        """This node does ___.

        Args:
            inputs (dict): Dictionary with keys "__", "__".

        Returns:
            outputs (dict): Dictionary with keys "__".
        """
        img = inputs['img']
        bboxes = inputs['bboxes']
        keypoints = inputs['keypoints']
        keypoint_scores = inputs['keypoint_scores']
        keypoint_conns = inputs['keypoint_conns']
        is_dab_list = ['Waiting...' for _ in bboxes]
        for i, _ in enumerate(bboxes):
            # row, col, _ = img.shape
            # key = self.keypoints_id['left_shoulder']
            # cv2.putText(
            #     img,'right shoulder',(int(keypoints[i][key][0]*col), int(keypoints[i][key][1]*row)), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,255,255),1,cv2.LINE_AA
            #     )
            left_wrist = keypoints[i][self.keypoints_id['left_wrist']]
            right_wrist = keypoints[i][self.keypoints_id['right_wrist']]
            left_elbow = keypoints[i][self.keypoints_id['left_elbow']]
            right_elbow = keypoints[i][self.keypoints_id['right_elbow']]
            left_shoulder = keypoints[i][self.keypoints_id['left_shoulder']]
            right_shoulder = keypoints[i][self.keypoints_id['right_shoulder']]
            self.nose = keypoints[i][self.keypoints_id['nose']]
            self.left_eye = keypoints[i][self.keypoints_id['left_eye']]
            self.right_eye = keypoints[i][self.keypoints_id['right_eye']]
            
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
            "img": img,
            "bboxes": bboxes,
            "keypoints": keypoints,
            "keypoint_conns": keypoint_conns,
            "keypoint_scores": keypoint_scores,
            "obj_tags": is_dab_list
        }

        return outputs

    def _is_straight_arm(
        self, lower_arm_vec: np.ndarray, elbow: List[float], shoulder: List[float]
        ) -> Tuple[float, bool]:
        upper_arm_vec = [elbow - shoulder]
        straight_score = cosine_similarity([lower_arm_vec], upper_arm_vec).ravel()[0]
        if straight_score >= self.thresholds['straight_arm']:
            return straight_score
        else:
            return False
    
    def _is_bent_arm(
        self, lower_arm_vec: np.ndarray, elbow: List[float], shoulder: List[float]
    ) -> Tuple[float, bool]:
        upper_arm_vec = [shoulder - elbow]
        bent_score = cosine_similarity([lower_arm_vec], upper_arm_vec).ravel()[0]
        if bent_score >= self.thresholds['bent_arm']:
            return bent_score
        else:
            return False

    def _is_head_close_to_wrist_or_elbow(
        self, wrist: List[float], elbow: List[float]
    ) -> float:

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
        parallel_score = cosine_similarity([self.lower_left_arm_vec], [self.lower_right_arm_vec]).ravel()[0]
        if parallel_score >= self.thresholds['lower_arm_parallel']:
            return parallel_score
        return False

    def _are_keypoints_present(
        self, left_wrist,right_wrist, left_elbow, right_elbow, left_shoulder, right_shoulder
        ) -> bool:
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
        total = 0
        for k, v in self.score_weightage.items():
            total += v
        for k, v in self.score_weightage.items():
            self.score_weightage[k] /= total
