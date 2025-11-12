Purpose:
- This folder is a sandbox for learning PyTorch from scratch using the book “Deep Learning with PyTorch” (O’Reilly). Work happens in an isolated conda environment so it never conflicts with other projects.

Conda environment (Python 3.13):
- Create: `conda create -n "learn_pytorch" python=3.13`
- Activate: `conda activate learn_pytorch`
- Upgrade tooling: `python -m pip install --upgrade pip`

Install PyTorch:
- CPU only: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu`
- NVIDIA GPU (CUDA; recommended): `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129`
    - Note: PyTorch wheels bundle the CUDA runtime; use cu124 even if your system has CUDA 12.9.41. Ensure your NVIDIA driver is up to date.
    - Alternative (conda): `conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia`
- If installation fails due to missing Python 3.13 wheels, recreate the env with Python 3.12: `conda create -n "learn_pytorch" python=3.12`

Jupyter (optional):
- Install: `pip install jupyterlab ipykernel`
- Register kernel: `python -m ipykernel install --user --name learn_pytorch --display-name "Python (learn_pytorch)"`

Reproducibility:
- Export env: `conda env export --no-builds > environment.yml`
- Freeze pip deps: `pip freeze > requirements.txt`

Maintenance:
- Deactivate: `conda deactivate`
- Remove env: `conda remove -n learn_pytorch --all`

Notes:
- Keep experiments, notebooks, and chapter code inside this folder to maintain isolation.
- When the book pins versions, align them here for consistent results.