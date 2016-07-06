Overview
============
This is a One-Shot Learning Handwritten Character Classifer written in Python using the SciPy library. This is the code for the One-Shot Learning episode of Fresh Machine Learning on [Youtube](https://youtu.be/FIjy3lV_KJU). The code trains against a few examples of handwritten characters and then tries to classify characters correctly. The error rate is around 38%. If you want to try a state-of-the-art, better-than-human, one-shot learning library that you can apply to all sorts of data, check out [this](https://github.com/MaxwellRebo/PyBPL) repo. 

Dependencies
============

* Python - (https://www.python.org/downloads/)
* scipy - `pip install scipy`
* numpy `pip install numpy`

Use [pip](https://pypi.python.org/pypi/pip) to install any missing dependencies

Basic Usage
===========
Step 1 - Run the code! It'll train against the handwritten character samples in the `all_runs` folder and then test it's classification ability.
It should output an average error rate of around 38%.
```shell
python demo_classification.py
```

Credits
===========
Credit for this demo code goes to the authors of the original BPL paper, this was the baseline demo code they used to compare their novel (much better) Matlab results against. 
