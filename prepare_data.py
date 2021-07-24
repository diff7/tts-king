from omegaconf import OmegaConf

from fs_two.preprocessor.preprocessor import Preprocessor


if __name__ == "__main__":
    preprocess_config = OmegaConf.load("./config.yaml")["preprocess_config"]
    preprocessor = Preprocessor(preprocess_config)
    preprocessor.build_from_path()


# TODO
# 1. preprocess from train % valid
# 2. choose what to process / all or specify
# 3. dataloader
# 4. debug and run
