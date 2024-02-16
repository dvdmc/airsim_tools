from airsim_tools.save_dataset import AirsimSaver, AirsimSaverConfig
from airsim_tools.config.configs import config

if __name__ == "__main__":
    saver = AirsimSaver(config)
    saver.setup()
    saver.save()


