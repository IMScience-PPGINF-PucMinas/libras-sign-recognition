import pandas as pd
import numpy as np

from idk_name import compute_distances, get_sign_predicted


class SignRecorder(object):
    def __init__(self, reference_signs: pd.DataFrame, seq_len=50):
        # Variables for recording
        self.is_recording = False
        self.seq_len = seq_len

        # List of results stored each frame
        self.recorded_results = []

        # DataFrame storing the distances between the recorded sign & all the reference signs from the dataset
        self.reference_signs = reference_signs

    def record(self):
        """
        Initialize sign_distances & start recording
        """
        self.reference_signs["distance"].values[:] = 0
        self.is_recording = True

    def process_results(self, results) -> (str, bool):
        """
        If the SignRecorder is in the recording state:
            it stores the landmarks during seq_len frames and then computes the sign distances
        :param results: mediapipe output
        :return: Return the word predicted (blank text if there is no distances)
                & the recording state
        """
        if self.is_recording:
            if len(self.recorded_results) < self.seq_len:
                self.recorded_results.append(results)
            else:
                self.reference_signs = compute_distances(
                    self.recorded_results, self.reference_signs)
                print(self.reference_signs)

                # Reset the recording variables
                self.is_recording = False
                self.recorded_results = []

        if np.sum(self.reference_signs["distance"].values) == 0:
            return "", self.is_recording
        return get_sign_predicted(batch_size=1), self.is_recording
