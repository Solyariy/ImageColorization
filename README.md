# Team 0
# Image Colorization

### How to run

Run `pip install -r requirements.txt` before starting

Project config is in `src/setup/config.py`

Before training model consider running `src/training/benchmark.py` to get best fitting batch size 
and number of workers for your machine.

To train model run `src/training/training_pypeline.py`.
Trained models are saved every 5 epochs to `trained_models/`.

To get colorized image run `src/colorize_image.py` specifying model name 
and path to image(optional, will colorize random image if not specified).
Colorized images are saved to `predicted_data/`.
