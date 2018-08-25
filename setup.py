from distutils.core import setup

setup(
    name='evolving_networks',
    version='0.1',
    packages=['evolving_networks', 'evolving_networks/neat', 'evolving_networks/pytorch',
              'evolving_networks/regulations', 'evolving_networks/reporting', 'evolving_networks/speciation',
              'evolving_networks/neat/configurations', 'evolving_networks/neat/genome',
              'evolving_networks/neat/phenome', 'evolving_networks/neat/reproduction',
              'evolving_networks/neat/genome/genes', 'evolving_networks/neat/phenome/proteins',
              'evolving_networks/pytorch/configurations', 'evolving_networks/pytorch/genome',
              'evolving_networks/pytorch/phenome', 'evolving_networks/pytorch/reproduction',
              'evolving_networks/pytorch/genome/genes'],
    install_requires=[
        'numpy', 'torch'
    ],
    url='https://turingequations.com',
    license='GNU GENERAL PUBLIC LICENSE 3',
    author='mdalvi',
    author_email='milind.dalvi@turingequations.com',
    description=''
)
