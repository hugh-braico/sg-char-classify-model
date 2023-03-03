# Skullgirls Character Classifier Training

Repo that I use to train the image classification models for
[Skug Stamper](https://github.com/hugh-braico/skug-stamper). 

Using a small set of cropped ingame screenshots, generate a much
larger set of training data by applying small translations and blurs
to simulate variations that you might see when looking at Skullgirls
footage. Then train Tensorflow models on that training data such that
the model can correctly enumerate the character.

Install requirements:

```bash
python3 -m pip install -r requirements.txt
```

Create model for char1:

```bash
# Generate training and testing data from screenshots
python3 process_char1_data.py

# Train the model
python3 train_char1_model.py

# Test the model
python3 test_char1_model.py
```

Create model for char2/3:

```bash
# Generate training and testing data from screenshots
python3 process_char23_data.py

# Train the model
python3 train_char23_model.py

# Test the model
python3 test_char23_model.py
```

Copy generated models to `models/` in skug-stamper repo.
