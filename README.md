# CAPTCHA Solver

## Set Up

It is advised that one should set up a python environment to run the code. You can use `venv` for this.
```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```
## Directory Structure
```bash
.
├── README.md
├── dataset
│   ├── generate_easy.py
│   ├── generate_hard.py
│   ├── generate_bonus.py
```

## Source Code
https://pillow.readthedocs.io/en/stable/handbook/text-anchors.html - to generate datasets
https://pillow.readthedocs.io/en/stable/reference/ImageFont.html#PIL.ImageFont.FreeTypeFont.getlength - to align letters of different fonts in the hard_set