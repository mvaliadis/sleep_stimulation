from distutils.core import setup


setup(
    name='eego1020',
    version='0.0.1',
    description='Toolbox to convert eego amplifier data into LSL streams.',
    long_description='A Python Toolbox to stream eego with LSL',
    author='Robert Guggenberger',
    author_email='robert.guggenberger@uni-tuebingen.de',
    url='git@github.com:translationalneurosurgery/app-eego.git',
    download_url='git@github.com:translationalneurosurgery/app-eego.git',
    license='MIT',
    packages=['eego1020'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Healthcare Industry',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Information Technology',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Human Machine Interfaces',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
        'Topic :: Software Development :: Libraries',
        ]
)
