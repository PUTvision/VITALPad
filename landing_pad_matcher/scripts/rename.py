from pathlib import Path

rgb_dir = Path('/home/rivi/Datasets/Landing/17/rgb')
labels_dir = Path('/home/rivi/Datasets/Landing/17/seg')

for file_path in rgb_dir.iterdir():
    file_path.rename(rgb_dir / file_path.name.replace('..', '.'))

for file_path in labels_dir.iterdir():
    file_path.rename(labels_dir / file_path.name.replace('..', '.'))
