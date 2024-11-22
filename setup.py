from setuptools import setup, find_packages

setup(
    name='aprendizagem_reforco',
    version='1.0.0',
    description='Modelo de Aprendizagem por Reforço desenvolvido para o projeto2 da cadeira de IASC',
    author='Diogo Correia',
    author_email='diogo.f.correia@protonmail.com',
    url='https://github.com/diogocorreia01/aprendizagem_reforco.git',
    packages=find_packages(),
    install_requires=[
        'colorama==0.4.6',
        'setuptools==75.6.0'
    ],
    python_requires='>=3.12',
)