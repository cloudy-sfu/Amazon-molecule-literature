import site
import os
import shutil

site_packages_path = site.getsitepackages()[0]
shutil.copy('method_gcastle/patch/simulator.py', os.path.join(site_packages_path, 'castle/datasets/simulator.py'))
shutil.copy('method_gcastle/patch/backend_init.py', os.path.join(site_packages_path, 'castle/backend/__init__.py'))

# https://github.com/pytorch/pytorch/issues/111573
shutil.copy('method_gcastle/patch/dag_gnn.py', os.path.join(site_packages_path, 'castle/algorithms/gradient/dag_gnn/torch/dag_gnn.py'))
shutil.copy('method_gcastle/patch/adam.py', os.path.join(site_packages_path, 'castle/algorithms/gradient/dag_gnn/torch/adam.py'))
shutil.copy('method_gcastle/patch/golem.py', os.path.join(site_packages_path, 'castle/algorithms/gradient/notears/torch/golem.py'))
shutil.copy('method_gcastle/patch/gran_dag.py', os.path.join(site_packages_path, 'castle/algorithms/gradient/gran_dag/torch/gran_dag.py'))
shutil.copy('method_gcastle/patch/al_trainer.py', os.path.join(site_packages_path, 'castle/algorithms/gradient/mcsl/torch/trainers/al_trainer.py'))
shutil.copy('method_gcastle/patch/gae_al_trainer.py', os.path.join(site_packages_path, 'castle/algorithms/gradient/gae/torch/trainers/al_trainer.py'))
shutil.copy('method_gcastle/patch/actor_graph.py', os.path.join(site_packages_path, 'castle/algorithms/gradient/rl/torch/models/actor_graph.py'))
shutil.copy('method_gcastle/patch/corl.py', os.path.join(site_packages_path, 'castle/algorithms/gradient/corl/torch/corl.py'))

shutil.copy('method_gcastle/patch/gae.py', os.path.join(site_packages_path, 'castle/algorithms/gradient/gae/torch/gae.py'))
