# Neural Network-Based Mapping Transformation Function for Improved Change Detection from SAR Satellite Images

This repository includes code for the paper **Neural Network-Based Mapping Transformation Function for Improved Change Detection from SAR Satellite Images**.

## Install dependencies

```
pip install -r requirements.txt
```

## Train the neural network

```
python main.py \
        --train_data "/path/to/train/records/*.tfrecord.GZIP" \
        --val_data "/path/to/val/records/*.tfrecord.GZIP" \
        --mixed_precision \
        --no_checkpoints \
        --epochs 50
```

## Acknowledgements

This project was developed in the [Data for Utilisation - Leveraging digitalisation through modern artificial intelligence solutions and cybersecurity project of Jamk University of Applied Sciences](https://www.jamk.fi/fi/tutkimus-ja-kehitys/tki-projektit/tieto-tuottamaan-digitalisaation-hyodyntaminen-modernien-tekoalyratkaisujen-ja-kyberturvallisuuden)
