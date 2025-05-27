from distutils.core import setup


setup(
    name='sleep_stimulation',
    version='0.0.1',
    description='Closed-loop sleep stimulation toolbox to process data in real time via LSL stream interface.',
    long_description='A Python toolbox that allows for online interface with LSL streams. The toolbox also includes a sleep classification framework for training data for online closed-loop gateing. Lastly, the packagae contains functions and pipelines for offline preprocessing and analysis.',
    author='Michael Valiadis',
    author_email='mvaliadis2@gmail.com',
    packages=['sleepstim', 'sleepstim.core'],
    url="https://github.com/neuromti/sleep_stimulation.git",
    download_url="https://github.com/neuromti/sleep_stimulation.git",
    license="MIT",
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
