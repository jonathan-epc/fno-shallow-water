from nconfig import PROJECT_ROOT, get_config

config = get_config()

OUTPUT_FOLDER = str(PROJECT_ROOT / config.paths.telemac_dir / "res")
STEERING_FOLDER = str(PROJECT_ROOT / config.paths.telemac_dir / "cas")
PARAMETERS_FILE = str(PROJECT_ROOT / config.paths.data_dir / "parameters.csv")
