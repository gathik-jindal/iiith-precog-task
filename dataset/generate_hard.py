"""
Date: 2025-04-05
Author: Gathik Jindal
Description: This script generates the hard set for task 0.
    It creates a folder called hard_set and generates 100 images of random words.
    The mapping between the image name and the word is saved in a csv file.
    The images are 200x100 pixels in size, but font size is random between 20 and 50.
    All letters have the same size but varying fonts, with noisy or textured background.
    The words are randomly generated and need not have any meaning.
    The words have varying capitalization, color and have a length of 3 to 10 characters.

Sources: add_grain function is taken from GPT.
         https://pillow.readthedocs.io/en/stable/handbook/text-anchors.html
"""

from PIL import Image, ImageDraw, ImageFont
import random
import string
import os
import csv
import math


def add_grain(img: Image.Image):
    """
    Adds random noise (Grainy Texture) to the image.
    """
    pixels = img.load()
    intensity = 60

    width, height = img.size

    for x in range(width):
        for y in range(height):
            r, g, b = pixels[x, y]

            # Add random noise
            def noise(): return random.randint(-intensity, intensity)
            r = max(0, min(255, r + noise()))
            g = max(0, min(255, g + noise()))
            b = max(0, min(255, b + noise()))

            pixels[x, y] = (r, g, b)


def generate_word():
    """
    Generates a random word of length between 3 and 10 characters.
    The word is generated using random letters from the English alphabet.
    Random letters are capitalized and the rest are lowercase.
    """

    length = random.randint(3, 8)
    word = ''.join(random.choices(
        string.ascii_lowercase.join(string.ascii_uppercase), k=length))
    return word


def get_random_font():
    """
    Returns a random font from the fonts folder.
    """
    fonts = os.listdir("fonts")
    font_file = random.choice(fonts)
    return os.path.join("fonts", font_file)


print("Generating 100 hard images...")
os.makedirs("dataset/hard_set", exist_ok=True)
fh = open("dataset/hard_set.csv", "w", newline="")
csv_writer = csv.writer(fh)

for i in range(100):
    word = generate_word()
    fontsize = random.randint(20, 50)
    im = Image.new("RGB", (250, 100), "white")
    d = ImageDraw.Draw(im)

    add_grain(im)

    x = 10
    y = 50
    fonts = [ImageFont.truetype(get_random_font(), fontsize)
             for _ in range(len(word))]

    for j in range(len(word)):
        font = fonts[j]
        d.text((x, y), word[j], fill=(random.randint(1, 255), random.randint(
            1, 255), random.randint(1, 255)), anchor="lm", font=font)
        if (j < len(word) - 1):
            # adjusting for kerning
            x += math.ceil(max(fonts[j+1].getlength(
                word[j:j+2]) - fonts[j+1].getlength(word[j+1]), font.getlength(
                word[j:j+2]) - font.getlength(word[j+1])))

    num = str(i + 1).zfill(3)
    im.save(f"dataset/hard_set/{num}.png")
    csv_writer.writerow([f"{num}.png", word])
    im.close()

fh.close()
print("Done!")
