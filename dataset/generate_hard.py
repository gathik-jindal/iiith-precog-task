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
"""

from PIL import Image, ImageDraw, ImageFont
import random
import string
import os
import csv


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
    font = ImageFont.truetype(get_random_font(), random.randint(20, 50))
    word = generate_word()
    im = Image.new("RGB", (250, 100), "white")
    d = ImageDraw.Draw(im)
    d.text((125, 50), word, fill="black", anchor="mm", font=font)
    num = str(i + 1).zfill(3)
    im.save(f"dataset/hard_set/{num}.png")
    csv_writer.writerow([f"{num}.png", word])
    im.close()

fh.close()
print("Done!")
