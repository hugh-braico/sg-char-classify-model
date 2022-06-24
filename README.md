# Skullgirls Character Classifier Training

Quick and dirty tensorflow model training repo

Add raw data to raw_char1_data and raw_char2_data

- use the existing data as an example
- use 1280x720 good quality screenshots
- one for every palette for char1, but a single screenshot is ok for char2

Install requirements

```bash
python3 -m pip install -r requirements.txt
```

Create model for char1

```bash
# Generate training and testing data from screenshots
python3 process_char1_data.py

# Train the model
python3 train_char1_model.py

# Test the model
python3 test_char1_model.py
```

Create model for char2/3

```bash
# Generate training and testing data from screenshots
python3 process_char23_data.py

# Train the model
python3 train_char23_model.py

# Test the model
python3 test_char23_model.py
```

Copy generated models to `skug-stamper/models`