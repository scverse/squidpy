From the Vizgen website for MERFISH Mouse Brain Receptor Map data release:
https://info.vizgen.com/mouse-brain-map

## STalign setup

These notebooks use Squidpy's experimental STalign support, which requires JAX.

### Install with GPU JAX

Create and activate an environment first, then install the CUDA-enabled JAX build and Squidpy:

```bash
python -m pip install --upgrade pip
python -m pip install -U "jax[cuda12]"
python -m pip install -e ".[jax]"
```

If you are not installing from the Squidpy repository root, install Squidpy from PyPI instead:

```bash
python -m pip install --upgrade pip
python -m pip install -U "jax[cuda12]"
python -m pip install "squidpy[jax]"
```

### Notes

- GPU JAX support is primarily for Linux with a compatible NVIDIA GPU and CUDA 12.
- Match the JAX CUDA installation to your local driver and CUDA setup.
- If GPU installation fails, check the official JAX installation guide:
  https://docs.jax.dev/en/latest/installation.html
