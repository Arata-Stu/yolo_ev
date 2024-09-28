from setuptools import setup, find_packages

# requirements.txtの内容を読み込む
def parse_requirements(filename):
    with open(filename, "r") as file:
        return [line.strip() for line in file if line.strip() and not line.startswith("#")]


setup(
    name='yolo_ev',
    version='0.1.0',
    description='A package for processing event camera data',
    author='Your Name',
    author_email='your.email@example.com',
    packages=find_packages(),
    install_requires=parse_requirements("requirements.txt"),  
)
