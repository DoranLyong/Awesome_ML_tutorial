import logging
import os.path as osp 
import sys 

import coloredlogs

this_dir = osp.dirname(__file__)  # directory path of this file 

# _Start: add 'lib' directory to PYTHONPATH
lib_path = osp.join(this_dir, "..", "libs")

if lib_path not in sys.path:
    sys.path.insert(0, lib_path)

# _End: 
coloredlogs.install(level="INFO", fmt="%(asctime)s %(filename)s %(levelname)s %(message)s")
logging.info("lib directory is added in sys.path: {}".format(lib_path))