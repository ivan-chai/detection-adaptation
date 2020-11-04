from setuptools import setup, find_namespace_packages


setup(
    version="0.0.1",
    name="dalib",
    long_description="Face detection and domain adaptation.",
    url="https://github.com/ivan-chai/detection-adaptation",
    author="Gleb Zhilin, Konstantin Sukharev, Ivan Karpukhin",
    author_email="ctrl-shift@yandex.ru, k.sukharev@gmail.com, karpuhini@yandex.ru",
    packages=find_namespace_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "torch",
        "torchvision",
        "Pillow",
        "pytorch-lightning==1.0.3",
        "PyYAML",
        "scipy"
    ]
)
