# Counting Coins
This project evolved dynamically and contains - beside the functional code - also old and/or failed approaches.
See the "Commands" section for information on how to run the code.

## Dependencies
Our project has the following dependencies:
- matplotlib
- pillow
- tqdm

## Commands
To execute one of the following programs, execute the respective command from the root directory
- Scraper: `python src/cnn_dataset/scraper.py` - downloads the dataset of single coin images
- CNN dataset: `python src/cnn_dataset/file_loader_module.py` - based on the dataset of single coin images, prepare CNN training data. This is only necessary if you want to train the CNN, if you already got weights this step is optional
- Retrieval database: `python src/retrieval_algorithm.py generate` - will generate a database of retrieval images. Use `--help` to see more options 
- Retrieval: `python src/retrieval_algorithm.py run --input <filename>` - find an image similar to `filename` w.r.t. to the sum of coin values
