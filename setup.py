import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
with open('VERSION') as version_file:
    version = version_file.read().strip()

setuptools.setup(
    name='intelli-recon',
    version=version,  # TODO add version to init
    author='Mher Khachatryan',
    author_email='mher.khachatryan4@edu.ysu.am',
    description='Intelli Recon, ASDS21 CV',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/mherkhachatryan/intelli-recon',
    license='',
    packages=['recon'],
    install_requires=['matplotlib>=3.1.1',
                      "numpy==1.21.0",
                      "pandas==1.5.2",
                      "Pillow==9.4.0",
                      "scikit_learn==1.2.0",
                      "segmentation_models_pytorch==0.3.1",
                      "torch==1.13.0",
                      "torchmetrics==0.11.0",
                      "tqdm==4.64.1",
                      "tensorboard==2.9"
                      ],
    python_requires='==3.8.16'
)
