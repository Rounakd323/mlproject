from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path: str) -> List[str]:
    """
    Return a list of requirements from requirements.txt
    """
    HYPHEN_E_DOT = '-e .'
    with open(file_path, "r") as file_obj:
        requirements = file_obj.readlines()
    requirements = [req.strip() for req in requirements if req.strip()]
    if HYPHEN_E_DOT in requirements:
        requirements.remove(HYPHEN_E_DOT)
    return requirements

setup(
    name='mlproject',
    version="0.0.1",
    author='Rounak',
    author_email='rounakd232@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
