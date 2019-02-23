from distutils.core import setup


setup(
    name='bnmetamodel_gh',
    version='0.0.1',
    packages=['bnmetamodel_gh'],
    url='',
    license='',
    author='Zack Xuereb Conti',
    author_email='zackxconti@gmail.com',
    description='bnmetamodel lib simplified for Lab Mouse grasshopper plugin',
    install_requires=['libpgm == 1.3', 'pybbn == 0.1.4', 'scipy == 0.18.1', 'numpy == 1.13.3', 'pandas == 0.19.0', 'sklearn == 0.0' ]
)
