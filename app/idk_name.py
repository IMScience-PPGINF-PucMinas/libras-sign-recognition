import array
import time
from collections import Counter
import pandas as pd

from utils.dtw import dtw_distances
from models.sign_model import SignModel
from utils.landmark_utils import extract_landmarks


def evaluate(recorded_results, reference_signs: pd.DataFrame, print_results=False):
    # Compute sign similarity with DTW (ascending order)
    updated_reference_signs = dtw_distances(recorded_results, reference_signs.copy())

    if print_results:
        print(updated_reference_signs.copy()[['name', 'signer', 'distance', 'video_id']].head(8))

    return get_sign_predicted(updated_reference_signs, batch_size=1)


def compute_distances(recorded_results: array.array, reference_signs: pd.DataFrame):
    """
    Updates the distance column of the reference_signs
    and resets recording variables
    """
    pose_list, left_hand_list, right_hand_list = [], [], []
    for results in recorded_results:
        pose, left_hand, right_hand = extract_landmarks(results)
        pose_list.append(pose)
        left_hand_list.append(left_hand)
        right_hand_list.append(right_hand)

    # Create a SignModel object with the landmarks gathered during recording
    recorded_sign = SignModel(pose_list, left_hand_list, right_hand_list)

    startTime = time.time()
    # Compute sign similarity with DTW (ascending order)
    updated_reference_signs = dtw_distances(recorded_sign, reference_signs)
    endTime = time.time()

    print(f"*** Time to compute distances: {endTime - startTime} seconds")

    return updated_reference_signs


def get_sign_predicted(reference_signs, batch_size=5, threshold=0.5):
    """
    Method that outputs the sign that appears the most in the list of closest
    reference signs, only if its proportion within the batch is greater than the threshold

    :param batch_size: Size of the batch of reference signs that will be compared to the recorded sign
    :param threshold: If the proportion of the most represented sign in the batch is greater than threshold,
                    we output the sign_name
                      If not,
                    we output "Sign not found"
    :return: The name of the predicted sign
    """
    # Get the list (of size batch_size) of the most similar reference signs
    sign_names = reference_signs.iloc[:batch_size]["name"].values

    # Count the occurrences of each sign and sort them by descending order
    sign_counter = Counter(sign_names).most_common()

    predicted_sign, count = sign_counter[0]
    if count / batch_size < threshold:
        return "Signe inconnu"
    return predicted_sign
