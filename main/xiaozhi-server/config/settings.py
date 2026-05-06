import os
from config.config_loader import read_config, get_project_dir, get_default_config_path, load_config


default_config_file = "config.yaml"
config_file_valid = False


def check_config_file():
    global config_file_valid
    if config_file_valid:
        return
    """
    简化的配置检查，仅提示用户配置文件的使用情况
    """
    custom_config_file = get_project_dir() + "data/." + default_config_file
    fallback_config_file = get_default_config_path()
    custom_config_exists = os.path.exists(custom_config_file)
    fallback_config_exists = os.path.exists(fallback_config_file)

    if not custom_config_exists and not fallback_config_exists:
        raise FileNotFoundError(
            "找不到data/.config.yaml、config.yaml或config_back.yaml文件，请确认至少存在一个配置文件"
        )

    source_config_file = (
        custom_config_file if custom_config_exists else fallback_config_file
    )

    # 检查是否从API读取配置
    config = load_config()
    if config.get("read_config_from_api", False):
        print("从API读取配置")
        old_config_origin = read_config(source_config_file)
        if old_config_origin.get("selected_module") is not None:
            error_msg = "您的配置文件好像既包含智控台的配置又包含本地配置：\n"
            error_msg += "\n建议您：\n"
            error_msg += "1、将根目录的config_from_api.yaml文件复制到data下，重命名为.config.yaml\n"
            error_msg += "2、按教程配置好接口地址和密钥\n"
            raise ValueError(error_msg)
    config_file_valid = True
