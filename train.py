import hydra
from omegaconf import DictConfig

from landing_pad_matcher.cli.training.density_estimator import train


@hydra.main(config_path='configs/', config_name='density_estimator.yaml')
def main(config: DictConfig) -> None:
    hydra.utils.get_original_cwd()

    train(**config)


if __name__ == '__main__':
    main()

