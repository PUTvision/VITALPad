import random
from pathlib import Path
from typing import List

import labelbox.exceptions
import numpy as np
from labelbox import DataRow, Client, Dataset, OntologyBuilder, Review

import cv2
from labelbox.data.annotation_types import Label, Geometry, MaskData, Mask


def main():
    path = Path('/home/rivi/Documents/PUT/Nagrania - landing pad')
    for recording_path in path.iterdir():
        output_dir = path / recording_path.stem
        output_dir.mkdir(exist_ok=True)

        video = cv2.VideoCapture(str(recording_path))
        fps = video.get(cv2.CAP_PROP_FPS)
        frames_to_skip = round(fps)

        frame_id = 0
        success = True
        while success:
            for _ in range(frames_to_skip):
                video.grab()
                frame_id += 1

            success, frame = video.read()
            if not success:
                continue

            cv2.imwrite(str(output_dir / f'{frame_id:05}.jpg'), frame)
            frame_id += 1


def upload():
    client = Client(api_key='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjbDBnc2gxMHk1b2cwMTAzZDlrOTcwNmI5Iiwib3JnYW5pemF0aW9uSWQiOiJjbDBnc2gxMGE1b2Z6MTAzZDNiMnA4em15IiwiYXBpS2V5SWQiOiJjbDB4dHRzeTEzdGtiMHo1dTVucWNlYWpuIiwic2VjcmV0IjoiNzljYmQ0NmEyMjA1YjJhY2ZlOTgxZDYyNTJlM2YzMjIiLCJpYXQiOjE2NDc2OTI5MjksImV4cCI6MjI3ODg0NDkyOX0.ktxsC-YiOREt0GOHVhtEfIoWLMCvFVgBpHUnGlafuCw')
    dataset = client.get_dataset('cl0xwct6f3vwu10bc88rahk1c')
    # dataset = client.create_dataset(name='LandingPad')
    path = Path('/home/rivi/Documents/PUT/Nagrania - landing pad')
    rows = []
    for images_dir in path.iterdir():
        if not images_dir.is_dir():
            continue

        for image_path in images_dir.iterdir():
            rows.append(dict(external_id=f'{images_dir.name}_{image_path.stem}', row_data=str(image_path)))

    random.shuffle(rows)
    to_retry = set()
    for row in rows:
        try:
            dataset.data_rows_for_external_id(row['external_id'])
        except labelbox.exceptions.ResourceNotFoundError:
            try:
                dataset.create_data_row(**row)
            except:
                to_retry.add(row)
        except:
            to_retry.add(row)

    for row in to_retry:
        try:
            dataset.data_rows_for_external_id(row['external_id'])
        except labelbox.exceptions.ResourceNotFoundError:
            try:
                dataset.create_data_row(**row)
            except:
                print(row)
        except:
            print(row)


def download():
    client = Client(api_key='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjbDBnc2gxMHk1b2cwMTAzZDlrOTcwNmI5Iiwib3JnYW5pemF0aW9uSWQiOiJjbDBnc2gxMGE1b2Z6MTAzZDNiMnA4em15IiwiYXBpS2V5SWQiOiJjbDB4dHRzeTEzdGtiMHo1dTVucWNlYWpuIiwic2VjcmV0IjoiNzljYmQ0NmEyMjA1YjJhY2ZlOTgxZDYyNTJlM2YzMjIiLCJpYXQiOjE2NDc2OTI5MjksImV4cCI6MjI3ODg0NDkyOX0.ktxsC-YiOREt0GOHVhtEfIoWLMCvFVgBpHUnGlafuCw')
    project = client.get_project('cl0xrk7bb3jgt10bc004mfzm7')
    labels = project.label_generator()

    rgb_dir = Path('rgb')
    labels_dir = Path('labels')

    rgb_dir.mkdir(exist_ok=True)
    labels_dir.mkdir(exist_ok=True)

    # Grab the first label and corresponding image
    for label in labels:
        reviews: List[Review] = label.extra['Reviews']
        score = 0
        for review in reviews:
            score += review.score

        if score < 1:
            continue

        rgb_data = cv2.cvtColor(label.data.value, cv2.COLOR_RGB2BGR)
        mask_data = np.zeros_like(rgb_data)
        for annotation in label.annotations:
            annotation_data = annotation.value.mask.value
            if annotation.name == 'Pad':
                mask_data = np.where(annotation_data == (255, 255, 255), (0, 255, 0), mask_data)
            else:
                mask_data = np.where(annotation_data == (255, 255, 255), (0, 0, 255), mask_data)

        cv2.imwrite(str(rgb_dir / f'{label.uid}.jpg'), rgb_data)
        if mask_data is not None:
            cv2.imwrite(str(labels_dir / f'{label.uid}.png'), mask_data)


if __name__ == '__main__':
    download()
