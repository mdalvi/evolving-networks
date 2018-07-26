from distutils.core import setup

setup(
    name='evolving_networks',
    version='0.01',
    packages=['notebooks', 'evolving_networks', 'evolving_networks/genome', 'evolving_networks/genome/genes',
              'evolving_networks/speciation', 'evolving_networks/complexity_regulation',
              'evolving_networks/reproduction', 'evolving_networks/phenome',
              'evolving_networks/phenome/proteins', 'evolving_networks/activations'],
    install_requires=[
        'numpy', 'gym'
    ],
    url='https://turingequations.com',
    license='GNU GENERAL PUBLIC LICENSE 3',
    author='mdalvi',
    author_email='milind.dalvi@turingequations.com',
    description=''
)
