import shutil
from pathlib import Path

path = Path('/home/rivi/Textures')
for texture_dir in path.iterdir():
    texture_path = texture_dir / f'{texture_dir.name.replace("-JPG", "")}_Color.jpg'
    if texture_path.exists():
        texture_path.rename(path / texture_path.name)

    if texture_path.is_dir():
        shutil.rmtree(texture_dir)
