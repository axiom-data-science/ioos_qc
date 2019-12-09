from setuptools import setup, find_packages


def version():
    with open('VERSION') as f:
        return f.read().strip()


def readme():
    with open('README.rst') as f:
        return f.read().strip()


reqs = [line.strip() for line in open('requirements.txt') if not line.startswith('#')]

setup(
    name                = 'ioos_qc_py2',
    version             = version(),
    description         = 'IOOS QARTOD and Quality Control tests implemented in Python',
    long_description    = readme(),
    license             = 'Apache',
    author              = "Kyle Wilcox",
    author_email        = "kyle@axds.co",
    url                 = "https://github.com/ioos/ioos_qc",
    packages            = find_packages(),
    install_requires    = reqs,
    classifiers         = [
        "Development Status :: 5 - Production/Stable",
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: GIS',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Physics'
    ]
)
