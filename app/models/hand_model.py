from typing import List

import numpy as np
import mediapipe as mp


class HandModel(object):
    """
    Params
        landmarks: List of positions
    Args
        connections: List of tuples containing the ids of the two landmarks representing a connection
        feature_vector: List of length 21 * 21 = 441 containing the angles between all connections
    """

    def __init__(self, landmarks: List[float]):

        # Define the connections
        self.connections = mp.solutions.holistic.HAND_CONNECTIONS

        self.landmarks = {
            "wrist": landmarks[0],
            "thumb_cmc": landmarks[1],
            "thumb_mcp": landmarks[2],
            "thumb_ip": landmarks[3],
            "thumb_tip": landmarks[4],
            "index_finger_mcp": landmarks[5],
            "index_finger_pip": landmarks[6],
            "index_finger_dip": landmarks[7],
            "index_finger_tip": landmarks[8],
            "middle_finger_mcp": landmarks[9],
            "middle_finger_pip": landmarks[10],
            "middle_finger_dip": landmarks[11],
            "middle_finger_tip": landmarks[12],
            "ring_finger_mcp": landmarks[13],
            "ring_finger_pip": landmarks[14],
            "ring_finger_dip": landmarks[15],
            "ring_finger_tip": landmarks[16],
            "pinky_mcp": landmarks[17],
            "pinky_pip": landmarks[18],
            "pinky_dip": landmarks[19],
            "pinky_tip": landmarks[20],
        }

        self.tracked_angles = [
            # 0, 1, 2
            ["wrist", "thumb_cmc", "thumb_mcp"],
            # 1, 2, 3
            ["thumb_cmc", "thumb_mcp", "thumb_ip"],
            # 2, 3, 4
            ["thumb_mcp", "thumb_ip", "thumb_tip"],
            # 1, 0, 5
            ["thumb_cmc", "wrist", "index_finger_mcp"],
            # 0, 5, 6
            ["wrist", "index_finger_mcp", "index_finger_pip"],
            # 5, 6, 7
            ["index_finger_mcp", "index_finger_pip", "index_finger_dip"],
            # 6, 7, 8
            ["index_finger_pip", "index_finger_dip", "index_finger_tip"],
            # 6, 5, 9
            ["index_finger_mcp", "wrist", "middle_finger_mcp"],
            # 0, 9, 10
            ["wrist", "middle_finger_mcp", "middle_finger_pip"],
            # 9, 10, 11
            ["middle_finger_mcp", "middle_finger_pip", "middle_finger_dip"],
            # 10, 11, 12
            ["middle_finger_pip", "middle_finger_dip", "middle_finger_tip"],
            # 10, 9, 13
            ["middle_finger_mcp", "wrist", "ring_finger_mcp"],
            # 0, 13, 14
            ["wrist", "ring_finger_mcp", "ring_finger_pip"],
            # 13, 14, 15
            ["ring_finger_mcp", "ring_finger_pip", "ring_finger_dip"],
            # 14, 15, 16
            ["ring_finger_pip", "ring_finger_dip", "ring_finger_tip"],
            # 14, 13, 17
            ["ring_finger_mcp", "wrist", "pinky_mcp"],
            # 0, 17, 18
            ["wrist", "pinky_mcp", "pinky_pip"],
            # 17, 18, 19
            ["pinky_mcp", "pinky_pip", "pinky_dip"],
            # 18, 19, 20
            ["pinky_pip", "pinky_dip", "pinky_tip"],
            # 18, 17, 13
            ["pinky_mcp", "wrist", "ring_finger_mcp"],
        ]

        self.feature_vector = self._get_feature_vector()

    def _get_feature_vector(self) -> List[float]:
        """
        Params
            landmarks: numpy array of shape (21, 3)
        Return
            List of length nb_connections * nb_connections containing
            the predefined angles between the connections
        """
        # connections = self._get_connections_from_landmarks(landmarks)

        angles_list = []

        for angle in self.tracked_angles:
            vector_a = self.landmarks[angle[0]] - self.landmarks[angle[1]]
            vector_b = self.landmarks[angle[2]] - self.landmarks[angle[1]]

            angle = self._get_angle_between_vectors(vector_a, vector_b)

            # If the angle is not NaN we store it else we store 0
            if angle == angle:
                angles_list.append(angle)
            else:
                angles_list.append(0)

        return angles_list

    @staticmethod
    def _get_angle_between_vectors(u: np.ndarray, v: np.ndarray) -> float:
        """
        https://www.cuemath.com/geometry/angle-between-vectors/
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
