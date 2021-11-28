import os
import pkg_resources
from setuptools import setup


PATH_ROOT = os.path.realpath(os.path.dirname(__file__))
REQUIREMENTS_PATH = os.path.join(PATH_ROOT, 'requirements.txt')


def parse_requirements(path: str):
    with open(path) as fp:
        text = fp.read()
    requirements = [
        str(requirement) for requirement
        in pkg_resources.parse_requirements(text)
    ]
    return requirements


setup(
    name='squeezer',
    version='0.1.0',
    packages=['squeezer'],
    url='https://github.com/esceptico/squeezer',
    license='MIT',
    author_email='ganiev.tmr@gmail.com',
    description='Lightweight knowledge distillation pipeline',
    install_requires=parse_requirements(REQUIREMENTS_PATH)
)
