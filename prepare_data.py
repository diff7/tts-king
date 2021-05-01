from omegaconf import OmegaConf

from fs_two.preprocessor.preprocessor import Preprocessor


if __name__ == "__main__":
    spreprocess_config = OmegaConf.load("./multi_config/config.yaml")[
        "preprocess"
    ]
    preprocessor = Preprocessor(spreprocess_config)
    preprocessor.build_from_path()
