from pathlib import Path
from dynaconf import Dynaconf

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.absolute()

settings = Dynaconf(
    envvar_prefix="FLPA",
    settings_files=[
        f"{PROJECT_ROOT}/config/settings.yaml",
    ],
    environments=True,
    env_switcher="FLPA_ENV",
    load_dotenv=True,
)
