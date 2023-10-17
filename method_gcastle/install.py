import site
import os
import shutil

site_packages_path = site.getsitepackages()[0]
shutil.copy('method_gcastle/patch/simulator.py', os.path.join(site_packages_path, 'castle/datasets/simulator.py'))
shutil.copy('method_gcastle/patch/backend_init.py', os.path.join(site_packages_path, 'castle/backend/__init__.py'))
