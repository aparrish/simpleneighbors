from setuptools import setup
    
with open('README.rst') as readme_file:
    readme = readme_file.read()
with open('HISTORY.rst') as history_file:
    history = history_file.read().replace('.. :changelog:', '')

setup(
    name='simpleneighbors',
    version='0.0.1',
    author='Allison Parrish',
    author_email='allison@decontextualize.com',
    url='https://github.com/aparrish/simpleneighbors',
    description='A clean and easy interface for nearest-neighbors lookup',
    long_description=readme + "\n\n" + history,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        'Programming Language :: Python',
        "License :: OSI Approved :: MIT License",
        "Topic :: Artistic Software",
        'Topic :: Software Development :: Libraries',
    ],
    package_dir={'simpleneighbors': 'simpleneighbors'},
    packages=['simpleneighbors'],
    install_requires=[
        'annoy'
    ],
    platforms='any',
    test_suite='tests'
)
