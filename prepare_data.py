from omegaconf import OmegaConf

from fs_two.preprocessor.preprocessor import Preprocessor


if __name__ == "__main__":
    preprocess_config = OmegaConf.load("./config.yaml")["preprocess"]
    preprocessor = Preprocessor(preprocess_config)
    preprocessor.build_from_path()
