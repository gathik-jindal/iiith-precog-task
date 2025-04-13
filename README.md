<a id="readme-top"></a>

<br />
<div align="center">

  <h1 align="center">CAPTCHA Solver Attempt</h1>

  <p align="center">
    An attempt to train a neural network (OCR) to crack CAPTCHAs
  </p>
</div>

---

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li><a href="#folder-structure">Folder Structure</a></li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>

## About The Project

This project is for IIITHs precog task. This project aims to capture an understanding at data generation, creating neural network and understanding the different layers to it.

### Built With

[![python][python]][python-url]
[![numpy][numpy]][numpy-url]
[![opencv][opencv]][opencv-url]
[![pytorch][pytorch]][pytorch-url]
[![sklearn][sklearn]][sklearn-url]
[![pillow][pillow]][pillow-url]

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Folder Structure

```bash
iiith-precog-task
|
|---dataset
      |---generate_easy.py
      |---generate_hard.py
      |---easy_set.csv
      |---hard_set.csv
      |---easy_set
            |---001.png
            ...
      |---hard_set
            |---001.png
            ...
|---fonts
      |---000_<font>
      ...
|---scripts
      |---classifier.py
      |---generator.py
      |---pre_processor_classification.py
      |---pre_processor_generator.py
|
|---README.md
|---REPORT.md
```

The `dataset` folder contains scripts and the datasets required for the tasks. This is the task 0 as required by the precog task.

The `fonts` folder contains 100 fonts that I webscrapped from `fonts.google.com`.

The `scripts` folder contains scripts for task 1 and task 2. It also contains the scripts required to `pre_processors` the data, i.e to make it in the desired structure for the neural networks to run.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Getting Started

### Prerequisites

This project requires python to installed on one's machine.

### Installation

Clone the repo, and install the required libraries.

```bash
# Navigate to a folder
git clone https://github.com/gathik-jindal/iiith-precog-task.git

# Create a virtual env
cd iiith-precog-task
python -m venv venv

# on linux
source venv/bin/activate
# on windows
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

To generate the datasets

```bash
python dataset/generate_easy.py
python dataset/generate_hard.py
```

To run the models (these scripts automatically call the `preprocessors`, so need to run them separately)
```bash
python scripts/classifier.py
python scripts/generator.py
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Acknowledgements

A lot of my code is inspired from [this](https://theailearner.com/2019/05/29/creating-a-crnn-model-to-recognize-text-in-an-image-part-1/) blog and the architecture was understood from [this](https://arxiv.org/pdf/1507.05717).

I would like to thank IIITH for giving me this opportunity to learn about OCRs and neural networks.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[python]: https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54
[python-url]: https://python.org
[numpy]: https://img.shields.io/badge/numpy-013243?style=for-the-badge&logo=numpy&logoColor=white
[numpy-url]: https://numpy.org
[opencv]: https://img.shields.io/badge/opencv-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white
[opencv-url]: https://opencv.org
[pytorch]: https://img.shields.io/badge/pytorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white
[pytorch-url]: https://pytorch.org
[sklearn]: https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white
[sklearn-url]: https://scikit-learn.org
[pillow]: https://img.shields.io/badge/pillow-DC643D?style=for-the-badge&logo=pillow&logoColor=white
[pillow-url]: https://python-pillow.org
