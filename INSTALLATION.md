## 🛠️ Installation

- Clone this repository:

  ```bash
  git clone https://github.com/EmbodiedGPT/EmbodiedGPT_Pytorch
  ```

- Create a conda virtual environment and activate it:

  ```bash
  conda create -n robohusky python=3.9.16 -y
  conda activate robohusky
  ```

- Install `PyTorch>=2.0` and `torchvision>=0.15.2` with `CUDA>=11.7`:

  For examples, to install `torch==2.0.1` with `CUDA==11.8`:

  ```bash
  conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
  # or
  pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
  ```

- Install `flash-attn`:

  ```bash
  git clone https://github.com/Dao-AILab/flash-attention.git
  cd flash-attention
  pip install flash-attn --no-build-isolation
  ```

- Install `transformers==4.34.1`:

  ```bash
  pip install transformers==4.34.1
  ```

- Install `apex` (optional):

  ```bash
  git clone https://github.com/NVIDIA/apex.git
  git checkout 2386a912164b0c5cfcd8be7a2b890fbac5607c82  # https://github.com/NVIDIA/apex/issues/1735
  pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
  ```

- Install other requirements:

  ```bash
  cd ..
  pip install -e EmbodiedGPT_Pytorch
  ```