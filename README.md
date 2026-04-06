# Brief

I am currently studying machine learning, so this repo is just another side project along my learning journey.

It includes data preprocessing code, and I plan to add model training code soon. The code is written using Marimo, so all `.py` files are actually notebooks.

Feel free to use whatever code you find here without attribution or anything like that.

The original data comes from Jeff Sackmann's tennis dataset repo, so make sure to give him credit. Big thanks to him.

# Notes

To get `input.csv` with the preprocessed data, run:

```
> ./fetch.sh
```

# Train

To run model training, execute:

```
uv run train_winner.py
```

But I recommend running it as a Marimo notebook instead:

```
uv run marimo edit train_winner.py
```
