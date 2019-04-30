
from models import create_models
from dataset_utils import load_dataset
from train import train_loop


DATASET = 'celeba'

BATCH_SIZE = 32
BASE_FEATURES = 64
LATENT_SIZE = 32
NUM_EPOCHS = 30

def init():
    data = load_dataset(DATASET, BATCH_SIZE)
    
    encoder_config = {
        'input_channels': 5,
        'base_features': BASE_FEATURES,
        'latent_size': LATENT_SIZE
    }

    decoder_config = {
        'input_channels': 10,
        'base_features': BASE_FEATURES,
        'output_channels': 3
    }
        
    encoder, decoder = create_models(DATASET, encoder_config, decoder_config)

    data_config = {
        'image_width': 128,
        'image_height': 128
    }

    config = {
        'batch_size': BATCH_SIZE,
        'latent_size': LATENT_SIZE,
        'num_epochs': NUM_EPOCHS,
        'data_config': data_config
    }

    
    train_loop(encoder, decoder, data, config)

if __name__ == "__main__":
    init()
