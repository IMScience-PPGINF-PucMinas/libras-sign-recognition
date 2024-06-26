import os

import pandas as pd
from tqdm import tqdm

from models.sign_model import SignModel
from utils.landmark_utils import save_landmarks_from_video, load_array


def load_dataset():
    videos = [
        file_name.replace(".mp4", "")
        for root, dirs, files in os.walk(os.path.join("app", "data", "videos"))
        for file_name in files
        if file_name.endswith(".mp4")
    ]
    dataset = [
        file_name.replace(".pickle", "").replace("pose_", "")
        for root, dirs, files in os.walk(os.path.join("app", "data", "dataset"))
        for file_name in files
        if file_name.endswith(".pickle") and file_name.startswith("pose_")
    ]

    # Create the dataset from the reference videos
    videos_not_in_dataset = list(set(videos).difference(set(dataset)))
    n = len(videos_not_in_dataset)
    if n > 0:
        print(f"\nExtracting landmarks from new videos: {n} videos detected\n")

        for idx in tqdm(range(n)):
            save_landmarks_from_video(videos_not_in_dataset[idx])

    return videos


def load_reference_signs(videos):
    reference_signs = {"name": [], "sign_model": [], "signer": [], "distance": [], "video_id": []}

    print("\nLoading reference signs\n")

    for video_name in tqdm(videos):
        sign_name, signer, _ = video_name.split("-")
        path = os.path.join("app", "data", "dataset", sign_name, video_name)

        pose_list = load_array(os.path.join(path, f"pose_{video_name}.pickle"))
        left_hand_list = load_array(os.path.join(path, f"lh_{video_name}.pickle"))
        right_hand_list = load_array(os.path.join(path, f"rh_{video_name}.pickle"))

        reference_signs["name"].append(sign_name)
        reference_signs["sign_model"].append(SignModel(pose_list, left_hand_list, right_hand_list))
        reference_signs['signer'].append(signer)
        reference_signs["distance"].append(0)
        reference_signs["video_id"].append(video_name)

    reference_signs = pd.DataFrame(reference_signs, dtype=object)
    print(
        f'\nDictionary count: {reference_signs[["name", "sign_model"]].groupby(["name"]).count()}\n'
    )
    return reference_signs
