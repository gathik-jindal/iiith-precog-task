# My journal on the Project

## Day 1

- **Date**: 2025-04-05
- **Tasks Completed**:
  - Set up the project repository.
  - Created initial project structure.
  - Wrote the README file.
  - Finished task 0. Wasn't very difficult.

I found this [link](https://pillow.readthedocs.io/en/stable/handbook/text-anchors.html), its precisely what I needed to draw the captchas.

- **Challenges**:
  - Figuring out how to get the letters printed in different fonts took longer than expected.
  - The noisy texture was very interesting, got too deep in it.

For Noisy texture, I used some AI and help of this link: [here](https://stackoverflow.com/questions/71818076/add-noise-with-varying-grain-size-in-python)

- **Improvements**
    - Its possible to use a better colouring function for the letters. They sometimes can be very light for humans to detect.
    - Its possible to first compute the lengths of the words and then decide starting x coordinate for the letters. This would have made the images a bit cleaner for the hard set.
    - Its possible to make the code much, much more optimized. But for just 100 images its not feasible. (Things like font loading, etc. can be done once and reused)

## Day 2 