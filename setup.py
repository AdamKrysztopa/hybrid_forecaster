from setuptools import setup, find_packages

setup(
    name='hybrid-forecaster',
    version='0.1.0',
    url='https://github.com/AdamKrysztopa/hybrid_forecaster',
    author='Adam Krysztopa',
    author_email='krysztopa@gmail.com',
    description='Testing package for hybrid forecasting joining statistical and machine learning methods',
    packages=find_packages(),    
    install_requires=[
        'numpy==1.23.5',
        'pandas==1.5.3',
        'plotly_express==0.4.1',
        'scipy==1.10.1',
        'scikit-learn==1.2.1',
        'statsforecast==1.5.0'
    ],
    python_requires='>=3.8.10',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
    ],
)