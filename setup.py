from distutils.core import setup

setup(
    name='evolving_networks',
    version='0.1',
    packages=['evolving_networks', 'evolving_networks/regulations', 'evolving_networks/reporting',
              'evolving_networks/speciation', 'evolving_networks/configurations', 'evolving_networks/genome',
              'evolving_networks/phenome', 'evolving_networks/reproduction', 'evolving_networks/genome/genes',
              'evolving_networks/phenome/proteins'],
    install_requires=['numpy', 'tabulate'],
    url='https://turingequations.com',
    license='GNU GENERAL PUBLIC LICENSE 3',
    author='mdalvi',
    author_email='milind.dalvi@turingequations.com',
    description=''
)
