BEGIN_SYMBOL = "⋅"
TERMINATE_SYMBOL = "○"

import re, string
import numpy as np
import pandas as pd
import os
from os.path import isfile
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras.backend as K
import csv

# Import custom modules
from Utils.EncoderModel import EncoderNetwork
from Utils.DecoderModel import DecoderNetwork
from Utils.AttentionMechanism import CustomAttention
from Utils.Seq2SeqModel import SequenceTransformer
from Utils.ConfigurationParams import ModelConfiguration
from Utils.DataProcessor import fetchDataset, locate_data, create_tokens, prepare_dataset

# Parse command line arguments
import argparse

def get_command_args():
    cmd_parser = argparse.ArgumentParser()
    cmd_parser.add_argument('--opt_algorithm', type=str, required=False)
    cmd_parser.add_argument('--learning_rate', type=float, required=False)
    cmd_parser.add_argument('--dropout_rate', type=float, required=False)
    cmd_parser.add_argument('--target_lang', type=str, required=False)
    cmd_parser.add_argument('--teacher_force_ratio', type=float, required=False)
    cmd_parser.add_argument('--embedding_dims', type=int, required=False)
    cmd_parser.add_argument('--training_epochs', type=int, required=False)
    cmd_parser.add_argument('--rnn_type', type=str, required=False)
    cmd_parser.add_argument('--encoder_count', type=int, required=False)
    cmd_parser.add_argument('--early_stop_patience', type=int, required=False)
    cmd_parser.add_argument('--decoder_count', type=int, required=False)
    cmd_parser.add_argument('--batch_size', type=int, required=False)
    cmd_parser.add_argument('--hidden_units', type=int, required=False)
    cmd_parser.add_argument('--use_attention', type=bool, required=False)
    cmd_parser.add_argument('--output_file', type=str, required=False)
    
    return cmd_parser.parse_args()

# Process command line arguments
def process_arguments(args):
    config = {
        'opt_algorithm': 'adam' if args.opt_algorithm is None else args.opt_algorithm,
        'target_lang': 'te' if args.target_lang is None else args.target_lang,
        'encoder_count': 1 if args.encoder_count is None else args.encoder_count,
        'batch_size': 128 if args.batch_size is None else args.batch_size,
        'embedding_dims': 64 if args.embedding_dims is None else args.embedding_dims,
        'rnn_type': 'lstm' if args.rnn_type is None else args.rnn_type,
        'early_stop_patience': 5 if args.early_stop_patience is None else args.early_stop_patience,
        'decoder_count': 1 if args.decoder_count is None else args.decoder_count,
        'hidden_units': 256 if args.hidden_units is None else args.hidden_units,
        'teacher_force_ratio': 1 if args.teacher_force_ratio is None else args.teacher_force_ratio,
        'use_attention': False if args.use_attention is None else args.use_attention,
        'learning_rate': 0.0005 if args.learning_rate is None else args.learning_rate,
        'training_epochs': 5 if args.training_epochs is None else args.training_epochs,
        'dropout_rate': 0.5 if args.dropout_rate is None else args.dropout_rate,
        'output_file': args.output_file
    }
    return config

# Evaluate model on test data
def evaluate_on_test_data(model, config, test_path):
    # Character-level evaluation
    test_data, _, _ = prepare_dataset(test_path, model.input_tokenizer, model.target_tokenizer)
    test_loss, test_accuracy = model.evaluate(test_data, batch_size=100)
    print(f'Character-level accuracy: {test_accuracy.numpy()}')

    # Word-level evaluation
    test_data_frame = pd.read_csv(test_path, sep="\t", header=None)
    input_words = test_data_frame[1].astype(str).tolist()
    target_words = test_data_frame[0].astype(str).tolist()
    
    predicted_outputs = []
    for word in input_words:
        predicted_outputs.append(model.translate(word)[0])

    word_accuracy = np.sum(np.array(predicted_outputs) == np.array(target_words)) / len(predicted_outputs)
    print(f"Word-level accuracy: {word_accuracy}")

    # Save results if specified
    if config['output_file'] is not None:
        results_df = pd.DataFrame({
            "inputs": input_words,
            "targets": target_words,
            "predictions": predicted_outputs
        })
        results_df.to_csv(config['output_file'])

    return model

def main():
    # Get and process command line arguments
    args = get_command_args()
    config = process_arguments(args)
    
    # Fetch dataset
    fetchDataset()
    
    # Get paths to data files
    train_path, validation_path, test_path = locate_data(config['target_lang'])
    
    # Prepare training data
    train_data, input_tokenizer, target_tokenizer = prepare_dataset(train_path)
    validation_data, _, _ = prepare_dataset(validation_path)
    
    # Create model configuration
    model_config = ModelConfiguration(
        language=config['target_lang'],
        embedding_dim=config['embedding_dims'],
        encoder_layers=config['encoder_count'],
        decoder_layers=config['decoder_count'],
        layer_type=config['rnn_type'],
        units=config['hidden_units'],
        dropout=config['dropout_rate'],
        epochs=config['training_epochs'],
        batch_size=config['batch_size'],
        attention=config['use_attention']
    )
    model_config.patience = config['early_stop_patience']
    model_config.save_outputs = config['output_file']
    
    # Build and train model
    transformer_model = SequenceTransformer(model_config)
    transformer_model.set_vocabulary(input_tokenizer, target_tokenizer)
    transformer_model.build(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metric=tf.keras.metrics.SparseCategoricalAccuracy(),
        optimizer=config['opt_algorithm'],
        lr=config['learning_rate']
    )
    
    # Train the model
    transformer_model.fit(
        train_data, 
        validation_data, 
        epochs=model_config.epochs, 
        wandb=None, 
        teacher_forcing_ratio=config['teacher_force_ratio']
    )
    
    # Evaluate model on test data
    evaluate_on_test_data(transformer_model, config, test_path)

if __name__ == "__main__":
    main()
