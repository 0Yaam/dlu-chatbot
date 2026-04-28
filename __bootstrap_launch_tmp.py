from pathlib import Path
from streamlit.web import bootstrap
main_script = str((Path.cwd() / 'dashboard.py').resolve())
flags = {
    'server_port': 8507,
    'server_headless': True,
    'server_fileWatcherType': 'none',
    'browser_gatherUsageStats': False,
    'browser_serverAddress': 'localhost',
}
bootstrap.load_config_options(flags)
bootstrap.run(main_script, False, [], flags)
