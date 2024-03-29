# Twissed

[![pypi version](https://img.shields.io/pypi/v/fbpic.svg)](https://pypi.python.org/pypi/fbpic)
[![License](https://img.shields.io/badge/License-MIT-blue)](LICENSE)
![Release](https://img.shields.io/github/release-date/Twissed/twissed
)

## Overview

Twissed is a Python package 🐍 that provides a large number of data analysis tools for **beam dynamics** and **laser wakefield acceleration** simulations.

Twissed is also a robust API between beam dynamics and laser-plasma simulation codes.

Twissed is multi-OS (Windows, Linux, Mac) and can be easily installed everywhere thanks to its low number of dependencies (numpy, pandas, scipy, h5py). It is especially made to run on HPC clusters (graphic packages, such as matplotlib are optional). 

![](https://twissed.github.io/_images/79f9ccfa14d8863894db9132e97ba6441951affc.png)

## Documentation and tutorials

The documentation is here: https://twissed.github.io

Tutorials are available here: https://github.com/Twissed/twissed_tutorial

Source code are here: https://github.com/Twissed/twissed

## API

Twissed can read and write data from several sources

* [FBPIC](https://fbpic.github.io/)
* [Smilei](https://smileipic.github.io/Smilei/)
* [Smilei via happi package](https://smileipic.github.io/Smilei/)
* [TraceWin](https://dacm-logiciels.fr/)
* [Astra]()

## Installation
### Requirement

Python 3.7 or greater

* Mandatory packages: numpy pandas scipy h5py. 
* Optional packages: matplotlib seaborn happi(smilei)

### Installation

* Mac OS/Linux

```shell
python -m pip install twissed
```

* Windows

```shell
py -m pip install twissed
```


#### It is also possible to download the source code for installation 

* Mac OS/Linux

```shell
git clone https://github.com/Twissed/twissed.git
python -m pip install -e .
```

* Windows

```shell
git clone https://github.com/Twissed/twissed.git
py -m pip install -e .
```
