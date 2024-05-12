import numpy as np


class PoseModel(object):
    def __init__(self, landmarks):

        self.landmarks = {
            "left_shoulder": landmarks[11],
            "right_shoulder": landmarks[12],
            "left_elbow": landmarks[13],
            "right_elbow": landmarks[14],
            "left_wrist": landmarks[15],
            "right_wrist": landmarks[16],
            "left_hip": landmarks[23],
            "right_hip": landmarks[24],
        }

        self.tracked_angles = [
            # 12, 11, 13
            ["right_shoulder", "left_shoulder", "left_elbow"],

            # 11, 13, 15
            ["left_shoulder", "left_elbow", "left_wrist"],

            # 13, 11, 23
            ["left_elbow", "left_shoulder", "left_hip"],

            # 11, 12, 14
            ["left_shoulder", "right_shoulder", "right_elbow"],

            # 12, 14, 16
            ["right_shoulder", "right_elbow", "right_wrist"],

            # 14, 12, 24
            ["right_elbow", "right_shoulder", "right_hip"],
        ]

        computed_angles = []

        for angle in self.tracked_angles:
            vector_a = self.landmarks[angle[0]] - self.landmarks[angle[1]]
            vector_b = self.landmarks[angle[2]] - self.landmarks[angle[1]]

            angle_result = self._get_angle_between_vectors(vector_a, vector_b)

            # If the angle is not NaN we store it else we store 0
            if angle_result == angle_result:
                computed_angles.append(angle_result)
            else:
                computed_angles.append(0)

        self.left_arm_embedding = computed_angles[0:3]
        self.right_arm_embedding = computed_angles[3:6]

    @staticmethod
    def _get_angle_between_vectors(u: np.ndarray, v: np.ndarray) -> float:
        """
        Args
            u, v: 3D vectors representing two connections
        Return
            Angle between the two vectors
        """
        if np.array_equal(u, v):
            return 0
        dot_product = np.dot(u, v)
        norm = np.linalg.norm(u) * np.linalg.norm(v)

        # TODO: we probably don't need to use the actual angle to compare
        return np.arccos(dot_product / norm)
