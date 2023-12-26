from phd_utils.airsim.save_dataset import AirsimSaver, AirsimSaverConfig
from phd_utils.airsim.configs import config

if __name__ == "__main__":
    saver = AirsimSaver(config)
    saver.setup()
    saver.save()


