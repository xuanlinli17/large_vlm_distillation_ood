from setuptools import setup, find_packages

def read_requirements():
    with open("requirements.txt", "r") as f:
        lines = [l.strip() for l in f.readlines()]
    install_requires = list(filter(None, lines))
    return install_requires
    
setup(
    name='large_vlm_distillation_ood',
    version='1.0.0',
    description="Implementation for the paper 'Distilling large vision-language models with out-of-distribution generalizability'.",
    author='Xuanlin Li, Yunhao Fang',
    packages=find_packages(include=['large_vlm_distillation_ood*']),
    install_requires=read_requirements(),
    python_requires=">=3.7",
)
