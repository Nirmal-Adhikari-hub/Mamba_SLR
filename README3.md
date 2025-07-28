### Step 1
```bash
pip install \
  torch==2.4.0+cu121 \
  torchvision==0.19.0+cu121 \
  torchaudio==2.4.0+cu121 \
  torch-tb-profiler==0.4.3 \
  torchmetrics==1.7.4 \
  --extra-index-url https://download.pytorch.org/whl/cu121
```

### Step 2
For making system use the toolkit from your conda environment rather than the **system's** one (which might be old to build the `causal-conv1d` package.)
```bash
# inside your (ch_slr) env:
conda install -y -c conda-forge \
   cudatoolkit=11.8 \
   nvcc_linux-64=11.8

# make sure this nvcc is first on your PATH:
export PATH=$CONDA_PREFIX/bin:$PATH

# verify
nvcc -V       # should now report 11.8.x

# now install everything (including causal-conv1d)
pip install -r pip_list.txt
```

