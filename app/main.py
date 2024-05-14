import cv2
import mediapipe
import pandas as pd
import questionary
from tqdm import tqdm

from utils.metrics_utils import compute_metrics
from idk_name import evaluate
from utils.dataset_utils import load_dataset, load_reference_signs
from utils.mediapipe_utils import mediapipe_detection
from sign_recorder import SignRecorder
from webcam_manager import WebcamManager


def online_evaluation(reference_signs: pd.DataFrame):
    # Object that stores mediapipe results and computes sign similarities
    sign_recorder = SignRecorder(reference_signs)

    # Object that draws keypoints & displays results
    webcam_manager = WebcamManager()

    # Turn on the webcam
    cap = cv2.VideoCapture(1, cv2.CAP_ANY)

    if not cap.isOpened():
        print("ERROR! Unable to open camera")
        exit

    # Set up the Mediapipe environment
    with mediapipe.solutions.holistic.Holistic(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as holistic:
        while cap.isOpened():

            # Read feed
            ret, frame = cap.read()

            # Make detections
            image, results = mediapipe_detection(frame, holistic)

            # Process results
            sign_detected, is_recording = sign_recorder.process_results(results)

            # Update the frame (draw landmarks & display result)
            webcam_manager.update(frame, results, sign_detected, is_recording)

            pressedKey = cv2.waitKey(1) & 0xFF
            if pressedKey == ord("r"):  # Record pressing r
                sign_recorder.record()
            elif pressedKey == ord("q"):  # Break pressing q
                break

        cap.release()
        cv2.destroyAllWindows()


def offline_evaluation(reference_signs: pd.DataFrame):
    # Define cross validation variables
    signers = reference_signs["signer"].unique()

    selected_signer = questionary.select(
        "Select a signer to use as the validation set", choices=signers).ask()

    training_set = reference_signs.loc[reference_signs["signer"] != selected_signer]
    validation_set = reference_signs.loc[reference_signs["signer"] == selected_signer]

    sign_pred = []
    sign_true = []

    signs_to_monitor = []

    #  Iterate over the validation set
    for _, row in tqdm(validation_set.iterrows(), total=validation_set.shape[0]):
        print_results = row['name'] in signs_to_monitor

        # Compute distance
        predicted_sign = evaluate(row['sign_model'], training_set, print_results=print_results)

        if print_results:
            print(f"Signer: {row['signer']}, Sign: {row['name']}, Video ID: {row['video_id']}")
            print(f"Predicted sign: {predicted_sign}")
            print("----------------------------------")

        sign_pred.append(predicted_sign)
        sign_true.append(row['name'])

    questionary.print(f"\n\nFinished cross validation for signer: {selected_signer}", style="bold")
    print(f"{len(training_set)} samples in the training set, {len(validation_set)} samples in the validation set")

    compute_metrics(sign_true, sign_pred)


if __name__ == "__main__":
    evaluation_mode = questionary.select(
        "Select the evaluation mode", choices=["ONLINE", "OFFLINE"]
    ).ask()

    # Create dataset of the videos where landmarks have not been extracted yet
    videos = load_dataset()

    # Create a DataFrame of reference signs (name: str, model: SignModel, distance: int)
    reference_signs = load_reference_signs(videos)

    if evaluation_mode == 'ONLINE':
        online_evaluation(reference_signs)
    elif evaluation_mode == 'OFFLINE':
        offline_evaluation(reference_signs)
