import tensorflow as tf
from metrics import masked_loss, masked_accuracy, optimizer
from model import create_transformer
from translator import standardize
from process_ds import get_processed_ds

if __name__ == "__main__":
    dataset = get_processed_ds().batch(64)
    for element in dataset:
        print(element)
    transformer = create_transformer()

    transformer.compile(
            loss = masked_loss,
            optimizer = optimizer,
            metrics = [masked_accuracy])

    transformer.fit(dataset, epochs=25)
