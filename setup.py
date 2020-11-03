import os

from setuptools import setup, find_namespace_packages

project_dir = os.path.dirname(os.path.realpath(__file__))

requirements_path = os.path.join(project_dir, "requirements.txt")
install_requires = []

if os.path.isfile(requirements_path):
    with open(requirements_path) as f:
        install_requires = f.read().splitlines()

setup(
    version="0.0.1",
    name="dalib",
    long_description="Face detection and domain adaptation.",
    url="https://github.com/ivan-chai/detection-adaptation",
    author="Gleb Zhilin, Konstantin Sukharev, Ivan Karpukhin",
    author_email="ctrl-shift@yandex.ru, k.sukharev@gmail.com, karpuhini@yandex.ru",
    packages=find_namespace_packages(where="src"),
    package_dir={"": "src"},
    install_requires=install_requires
)
