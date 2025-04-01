import os
import warnings

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from collections import Counter
import joblib
import json
import numpy as np
import pretty_midi
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate, Masking, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tqdm import tqdm

class MidiModel:
    def __init__(self, lstm_units, dropout_rate, l2_reg, sequence_length, dataset_path):
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.sequence_length = sequence_length
        self.instrument_to_layer = self.create_instrument_mapping(dataset_path)
        self.max_instruments = len(self.instrument_to_layer)
        self.model = self._build_multi_branch_model()
        self.scaler = MinMaxScaler()

    @staticmethod
    def create_instrument_mapping(dataset_path, top_n=10):
        # Check if the mapping file already exists
        mapping_file = 'saved_models/instrument_mapping/instrument_mapping.joblib'
        if os.path.exists(mapping_file):
            print("Loading existing instrument mapping...")
            return joblib.load(mapping_file)

        instrument_counter = Counter()
        all_tracks = 0
        skipped_tracks = 0
        total_unique_instruments = set()

        for artist in tqdm(os.listdir(dataset_path), desc="Creating instrument mapping"):
            artist_path = os.path.join(dataset_path, artist)
            if os.path.isdir(artist_path):
                for midi_file in os.listdir(artist_path):
                    all_tracks += 1
                    try:
                        midi_path = os.path.join(artist_path, midi_file)
                        midi_data = pretty_midi.PrettyMIDI(midi_path)
                        instruments_in_file = set()
                        for instrument in midi_data.instruments:
                            if not instrument.is_drum:  # Skip drums
                                instruments_in_file.add(instrument.program)
                                total_unique_instruments.add(instrument.program)
                        instrument_counter.update(instruments_in_file)
                    except Exception:
                        skipped_tracks += 1

        print(f"Skipped {skipped_tracks}/{all_tracks} tracks")

        # Select top N most common instruments, excluding drums
        top_instruments = [instr for instr, _ in instrument_counter.most_common(top_n)]
        
        # Print the names of the top N instruments out of total unique instruments
        print(f"Top {top_n} instruments out of {len(total_unique_instruments)}:")
        for program in top_instruments:
            instrument_name = pretty_midi.program_to_instrument_name(program)
            print(f"\tInstrument program {program}: {instrument_name}")

        # Sort and map each program number to an index
        sorted_programs = sorted(top_instruments)
        instrument_mapping = {program: i for i, program in enumerate(sorted_programs)}

        # Save the mapping
        save_path = os.path.dirname(mapping_file)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        joblib.dump(instrument_mapping, mapping_file)

        return instrument_mapping
    
    @staticmethod
    def get_artist_names(directory):
        return [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]
    
    @staticmethod
    def get_unique_instruments(dataset_path):
        unique_instruments = set()
        for artist in os.listdir(dataset_path):
            artist_path = os.path.join(dataset_path, artist)
            if os.path.isdir(artist_path):
                for midi_file in os.listdir(artist_path):
                    midi_path = os.path.join(artist_path, midi_file)
                    try:
                        midi_data = pretty_midi.PrettyMIDI(midi_path)
                        for instrument in midi_data.instruments:
                            unique_instruments.add(instrument.program)
                    except Exception as e:
                        pass
        return unique_instruments
    
    @staticmethod
    def is_valid_midi(midi_path):
        try:
            _ = pretty_midi.PrettyMIDI(midi_path)
            return True
        except Exception:
            return False

    def _build_multi_branch_model(self):
        print("Building multi-branch LSTM...")
        input_layers = []
        lstm_layers = []

        # Create individual LSTM layers for each instrument
        for _ in tqdm(range(self.max_instruments), desc="Creating LSTM layers"):
            input_layer = Input(shape=(self.sequence_length, 6))  # 6 features per timestep
            masking_layer = Masking(mask_value=0.0)(input_layer)
            lstm_layer = LSTM(self.lstm_units, return_sequences=True, 
                            kernel_regularizer=l2(self.l2_reg),
                            recurrent_regularizer=l2(self.l2_reg))(masking_layer)
            lstm_layer = Dropout(self.dropout_rate)(lstm_layer)
            lstm_layer = BatchNormalization()(lstm_layer)
            lstm_layers.append(lstm_layer)
            input_layers.append(input_layer)

        # Concatenate the outputs from individual LSTM layers
        concatenated = Concatenate()(lstm_layers)

        # Shared LSTM layer that processes the concatenated features
        shared_context = LSTM(self.lstm_units, return_sequences=False, 
                            kernel_regularizer=l2(self.l2_reg), 
                            recurrent_regularizer=l2(self.l2_reg))(concatenated)
        shared_context = Dropout(self.dropout_rate)(shared_context)
        shared_context = BatchNormalization()(shared_context)

        # Output layers for each instrument
        outputs = []
        for _ in range(self.max_instruments):
            output = Dense(4, activation='linear', 
                        kernel_regularizer=l2(self.l2_reg))(shared_context)
            outputs.append(output)

        model = tf.keras.Model(inputs=input_layers, outputs=outputs)
        model.compile(loss='mean_squared_error', optimizer=Adam())
        return model

    def _extract_features_and_labels_from_midi(self, midi_path, artist_id):
        try:
            midi_data = pretty_midi.PrettyMIDI(midi_path)
        except (OSError, IOError, EOFError, pretty_midi.PrettyMIDIError) as e:
            print(f"Error processing file {midi_path}: {e}")
            return {}, {}
        
        features_by_instrument = {}
        labels_by_instrument = {}

        for instrument in midi_data.instruments:
            if not instrument.is_drum and instrument.program in self.instrument_to_layer:
                instrument_features = []
                instrument_labels = []

                for i in range(len(instrument.notes) - self.sequence_length):
                    # Extract feature sequence
                    feature_sequence = [
                        [
                            note.pitch, note.velocity, note.end - note.start, 
                            note.start, instrument.program, artist_id
                        ] 
                        for note in instrument.notes[i:i + self.sequence_length]
                    ]

                    # Extract label (next note after the sequence)
                    next_note = instrument.notes[i + self.sequence_length]
                    label_sequence = [
                        next_note.pitch, next_note.velocity, 
                        next_note.end - next_note.start, next_note.start
                    ]

                    instrument_features.append(feature_sequence)
                    instrument_labels.append(label_sequence)

                if instrument_features:
                    features_by_instrument[instrument.program] = np.array(instrument_features, dtype=np.float32)
                    labels_by_instrument[instrument.program] = np.array(instrument_labels, dtype=np.float32)

        return features_by_instrument, labels_by_instrument

    def _reshape_into_sequences(self, features, sequence_length):
        sequences = []
        for i in range(0, len(features) - sequence_length + 1):
            sequences.append(features[i:i + sequence_length])
        return np.array(sequences, dtype=np.float32)

    def preprocess_data(self, dataset_path, artist_encoder):
        print("Preproccessing data...")
        all_features_by_instrument = {i: [] for i in range(self.max_instruments)}
        all_labels_by_instrument = {i: [] for i in range(self.max_instruments)}

        for artist in tqdm(os.listdir(dataset_path), desc="Processing artists"):
            artist_id = artist_encoder.transform([artist])[0]
            artist_path = os.path.join(dataset_path, artist)
            if os.path.isdir(artist_path):
                for midi_file in os.listdir(artist_path):
                    midi_path = os.path.join(artist_path, midi_file)
                    if not MidiModel.is_valid_midi(midi_path):
                        continue
                    features_by_instrument, labels_by_instrument = self._extract_features_and_labels_from_midi(midi_path, artist_id)

                    # Assigning features and labels to the appropriate layer based on the instrument mapping
                    for program_number, features in features_by_instrument.items():
                        layer_index = self.instrument_to_layer.get(program_number)
                        if layer_index is not None:
                            all_features_by_instrument[layer_index].extend(features)
                            all_labels_by_instrument[layer_index].extend(labels_by_instrument[program_number])

        for i in tqdm(range(self.max_instruments), desc="Padding sequences"):
            feature_array = np.array(all_features_by_instrument[i], dtype=np.float32)
            label_array = np.array(all_labels_by_instrument[i], dtype=np.float32)

            # Check if the array is empty
            if feature_array.size == 0:
                all_features_by_instrument[i] = np.zeros((1, self.sequence_length, 6), dtype=np.float32)
                all_labels_by_instrument[i] = np.zeros((1, 4), dtype=np.float32)  # Adjust label shape to (1, 4)
            else:
                # Reshape to 3D for features and 2D for labels (no reshape needed for labels if they are already correct)
                if feature_array.ndim == 2:
                    feature_array = feature_array.reshape(-1, self.sequence_length, 6)

                all_features_by_instrument[i] = feature_array
                all_labels_by_instrument[i] = label_array  # No reshape for labels

        return all_features_by_instrument, all_labels_by_instrument
    
    def scale_data(self, features_by_instrument, labels_by_instrument):
        print("Scaling data...")

        # Combine continuous features and labels from all instruments for scaling
        all_continuous_data = np.concatenate(
            [
                np.concatenate((features[:, :, :4].reshape(-1, 4), labels.reshape(-1, 4)), axis=0)
                for features, labels in zip(features_by_instrument.values(), labels_by_instrument.values())
                if features.size > 0 and labels.size > 0
            ],
            axis=0
        )

        # Fit the scaler on the combined continuous features and labels
        self.scaler.fit(all_continuous_data)

        # Save the scaler
        save_path = 'saved_models/scalers'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        joblib.dump(self.scaler, os.path.join(save_path, 'midi_scaler.joblib'))

        # Scale both features and labels for each instrument
        for instr in features_by_instrument.keys():
            if features_by_instrument[instr].size == 0 or labels_by_instrument[instr].size == 0:
                continue

            # Scale features
            continuous_features = features_by_instrument[instr][:, :, :4]
            num_samples, sequence_length, num_continuous_features = continuous_features.shape
            continuous_features_reshaped = continuous_features.reshape(-1, num_continuous_features)
            continuous_features_scaled = self.scaler.transform(continuous_features_reshaped)
            continuous_features_scaled = continuous_features_scaled.reshape(num_samples, sequence_length, num_continuous_features)
            categorical_features = features_by_instrument[instr][:, :, 4:]
            features_by_instrument[instr] = np.concatenate([continuous_features_scaled, categorical_features], axis=-1)

            # Scale labels
            labels = labels_by_instrument[instr]
            labels_reshaped = labels.reshape(-1, 4)
            labels_scaled = self.scaler.transform(labels_reshaped)
            labels_scaled = labels_scaled.reshape(-1, 4)
            labels_by_instrument[instr] = labels_scaled

        return features_by_instrument, labels_by_instrument
        
    def train(self, train_data, epochs, batch_size, validation_data=None, save_history=True):
        print("Training model...")
        X_train, y_train = train_data
        X_val, y_val = validation_data
        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                                validation_data=(X_val, y_val),
                                callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)])
        if save_history:
            self.save_training_history(history)
        return history
    
    def prepare_training_data(self, all_features_by_instrument, all_labels_by_instrument):
        print("Preparing training data...")
        min_samples = min([features.shape[0] for features in all_features_by_instrument.values() if features.shape[0] > 0])
        print(f"Min samples: {min_samples}")
        train_features = []
        train_labels = []

        for i in tqdm(range(self.max_instruments), "Preparing features by instrument"):
            if i in all_features_by_instrument and all_features_by_instrument[i].shape[0] > 0:
                features = all_features_by_instrument[i]
                labels = all_labels_by_instrument[i]

                # Truncate to match min_samples
                features = features[:min_samples]
                labels = labels[:min_samples]

                train_features.append(features)
                train_labels.append(labels)
            else:
                # Append zero arrays if an instrument is missing or has 0 samples
                train_features.append(np.zeros((min_samples, self.sequence_length, 6), dtype=np.float32))
                train_labels.append(np.zeros((min_samples, 4), dtype=np.float32))  # Labels have a shape of (min_samples, 4)

        train_features = [features.astype('float32') for features in train_features]
        train_labels = [labels.astype('float32') for labels in train_labels]
        return train_features, train_labels
    
    def convert_seed_to_midi(self, seed_sequences, artist_id):
        sequence_to_midi = {instr: seed_sequences.get(layer_index, []) for instr, layer_index in self.instrument_to_layer.items()}
        for instr in sequence_to_midi:
            processed_seed_sequence = self.post_process_sequence(sequence_to_midi[instr], instr, artist_id)
            sequence_to_midi[instr] = processed_seed_sequence
        seed_as_midi = self.convert_to_midi(sequence_to_midi)
        self.save_music(seed_as_midi, "seed_sequence")

    def generate_seed_sequence(self, artist_id):
        print("Generating seed sequence...")
        np.random.seed(42)
        seed_sequences = {}
        if not hasattr(self.scaler, 'n_features_in_'):
            self.scaler = joblib.load('saved_models/scalers/midi_scaler.joblib')

        for program, layer_index in self.instrument_to_layer.items():
            seed_sequence = np.zeros((self.sequence_length, 6))  # 6 features per timestep
            seed_sequence[:, 0] = np.random.randint(0, 127, self.sequence_length)  # Pitch
            seed_sequence[:, 1] = np.random.randint(0, 127, self.sequence_length)  # Velocity
            seed_sequence[:, 2] = np.full(self.sequence_length, 0.5)  # Duration
            seed_sequence[:, 3] = np.arange(0, self.sequence_length * 0.5, 0.5)  # Start time
            seed_sequence[:, 4] = program  # Instrument program (categorical)
            seed_sequence[:, 5] = artist_id  # Artist ID (categorical)

            continuous_features_scaled = self.scaler.transform(seed_sequence[:, :4])
            seed_sequence[:, :4] = continuous_features_scaled
            seed_sequences[layer_index] = seed_sequence

        self.convert_seed_to_midi(seed_sequences, artist_id)

        seed_sequence_list = [seed_sequences[layer_index] for layer_index in range(self.max_instruments)]
        for i in range(len(seed_sequence_list)):
            seed_sequence_list[i] = seed_sequence_list[i].reshape((1, self.sequence_length, 6))
        
        return seed_sequence_list

    def generate_music(self, artist_encoder, artist_name):
        artist_id = artist_encoder.transform([artist_name])[0]
        seed_sequence = self.generate_seed_sequence(artist_id)
        generated_music = {index: [] for _, index in self.instrument_to_layer.items()}
        layer_to_instrument = {index: program for program, index in self.instrument_to_layer.items()}

        for _ in range(self.sequence_length):
            predictions = self.model.predict(seed_sequence)
            new_seed_sequence = []
            for layer_index, prediction in enumerate(predictions):
                prediction = np.append(prediction, [layer_to_instrument.get(layer_index), artist_id])
                generated_music[layer_index].append(prediction)
                updated_sequence = seed_sequence[layer_index][:, 1:, :]
                new_prediction = prediction[np.newaxis, np.newaxis, :]
                updated_sequence = np.concatenate([updated_sequence, new_prediction], axis=1)
                new_seed_sequence.append(updated_sequence)
            seed_sequence = new_seed_sequence

        for instr in generated_music:
            processed_sequence = self.post_process_sequence(np.vstack(generated_music[instr]), instr, artist_id)
            generated_music[instr] = processed_sequence

        return self.convert_to_midi(generated_music)

    def post_process_sequence(self, sequence, instrument_program, artist_id):
        processed_sequence = []
        features_to_inverse_transform = np.array(sequence)[:, :4]
        inverse_transformed_features = self.scaler.inverse_transform(features_to_inverse_transform)

        for idx, note in enumerate(sequence):
            inverse_transformed_note = inverse_transformed_features[idx]

            processed_note = [
                max(0, min(127, int(round(inverse_transformed_note[0])))),   # Pitch
                max(0, min(127, int(round(inverse_transformed_note[1])))),   # Velocity
                max(0, inverse_transformed_note[2]),                         # Duration
                max(0, inverse_transformed_note[3]),                         # Start time
                instrument_program,
                artist_id
            ]
            processed_sequence.append(processed_note)
        return processed_sequence

    def split_instrument_data(self, train_features, train_labels):
        print("Splitting data into train and validation sets...")
        excluded_count = 0
        X_train_list, X_val_list, y_train_list, y_val_list = [], [], [], []

        for features, labels in zip(train_features, train_labels):
            if len(features) >= 2:  # Ensure there are at least two samples
                # No shuffling as this is sequential data
                X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.2, shuffle=False)
                X_train_list.append(X_train)
                X_val_list.append(X_val)
                y_train_list.append(y_train)
                y_val_list.append(y_val)
            else:
                excluded_count += 1

        print(f"Excluded {excluded_count} datasets due to insufficient samples.")
        return X_train_list, y_train_list, X_val_list, y_val_list

    def convert_to_midi(self, generated_music):
        print("Converting to MIDI...")
        generated_midi = pretty_midi.PrettyMIDI()
        for instr, music in generated_music.items():
            instrument = pretty_midi.Instrument(program=instr)
            for note_features in music:
                # Extract note features
                pitch, velocity, duration, start_time, _, _ = note_features
                end_time = start_time + duration
                note = pretty_midi.Note(velocity=int(velocity), pitch=int(pitch), start=start_time, end=end_time)
                instrument.notes.append(note)
            generated_midi.instruments.append(instrument)
        return generated_midi
    
    def save_model(self, file_path=None):
        print("Saving model...")
        default_save_dir = 'saved_models/midi_models'  
        default_save_path = os.path.join(default_save_dir, 'lstm_model.keras')
        save_path = file_path if file_path else default_save_path
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.model.save(save_path)
        print(f"Model saved in {save_path}")

    def load_model(self, file_path):
        print("Loading model...")
        self.model = tf.keras.models.load_model(file_path)

    def save_music(self, midi_object, artist, filename=None):
        print("Saving music...")
        output_dir = 'generated_music'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        artist_filename = "generated_" + artist.replace(" ", "_").lower()
        if filename is None:
            filename = f'{artist_filename}.mid'
        output_path = os.path.join(output_dir, filename)
        midi_object.write(output_path)
        print(f"Generated audio saved in {output_path}")
    
    def save_training_history(self, history):
        history_dict = history.history
        with open('plots/training_history.json', 'w') as file:
            json.dump(history_dict, file) 
    
    def grid_search(self, param_grid, train_data, val_data, epochs, batch_size, save_dir='grid_search_results'):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        best_performance = float('inf')
        best_model_params = None

        for lstm_units in param_grid['lstm_units']:
            print(f"Training with: lstm_units={lstm_units}")
            self.lstm_units = lstm_units
            self.model = self._build_multi_branch_model()

            history = self.train(train_data, epochs, batch_size, validation_data=val_data, save_history=False)
            val_loss = min(history.history['val_loss'])
            
            # Save model and history
            model_filename = f"model_units_{lstm_units}.keras"
            history_filename = f"history_units_{lstm_units}.json"
            self.save_model(os.path.join(save_dir, model_filename))
            with open(os.path.join(save_dir, history_filename), 'w') as file:
                json.dump(history.history, file)

            # Update best model
            if val_loss < best_performance:
                best_performance = val_loss
                best_model_params = lstm_units

        # Load and return the best model
        if best_model_params:
            print(f"Best model: LSTM units: {best_model_params}")
            best_model_path = f"model_units_{best_model_params}.keras"
            self.load_model(os.path.join(save_dir, best_model_path))
            return self.model

def main():
    dataset_path = 'datasets/clean_midi'
    sequence_length = 50
    lstm_units = 256
    dropout_rate = 0.3
    l2_reg = 0.01
    epochs = 1000
    batch_size = 128
    model_save_path = 'saved_models/midi_models/lstm_model.keras' 

    param_grid = {
        'lstm_units': [256, 512]
    }

    print("Initializing model...")
    midi_model = MidiModel(lstm_units, dropout_rate, l2_reg, sequence_length, dataset_path)
    
    print("Initializing artist encoder...")
    artist_encoder = LabelEncoder()
    artist_encoder.fit(MidiModel.get_artist_names(dataset_path))
    
    if os.path.exists(model_save_path):
        midi_model.load_model(model_save_path)
    else:
        all_features_by_instrument, all_labels_by_instrument = midi_model.preprocess_data(dataset_path, artist_encoder)
        all_features_by_instrument, all_labels_by_instrument = midi_model.scale_data(all_features_by_instrument, all_labels_by_instrument)
        train_features, train_labels = midi_model.prepare_training_data(all_features_by_instrument, all_labels_by_instrument)
        X_train_list, y_train_list, X_val_list, y_val_list = midi_model.split_instrument_data(train_features, train_labels)
        # midi_model.train((X_train_list, y_train_list), epochs, batch_size, validation_data=(X_val_list, y_val_list))
        # midi_model.save_model(model_save_path)
        best_model = midi_model.grid_search(param_grid, (X_train_list, y_train_list), (X_val_list, y_val_list), epochs, batch_size)

    artist_name = 'The Beatles' 
    generated_midi = midi_model.generate_music(artist_encoder, artist_name)
    midi_model.save_music(generated_midi, artist_name)