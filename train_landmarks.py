import hydra
from omegaconf import DictConfig

from landing_pad_matcher.cli.training.landmarks_matcher import train


@hydra.main(config_path='configs/', config_name='landmarks_matcher.yaml')
def main(config: DictConfig) -> None:
    train(**config)


if __name__ == '__main__':
    main()

