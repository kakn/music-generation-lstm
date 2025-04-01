# Music Generation with LSTM Neural Networks

This project represents the intersection of music and artificial intelligence, two of my hobbies, developed as part of my Master's in Computer Science. 
The code generates original music by training LSTM networks on MIDI files from different artists.

## Features
- Multi-branch LSTM model that learns instrument-specific patterns
- Handles multiple instruments simultaneously
- Processes pitch, velocity, duration, and timing from MIDI files
- Generates new compositions in the style of trained artists

## Usage
1. Place MIDI files in `datasets/clean_midi/{artist_name}/`
2. Run the model:
   ```
   python src/multi_branch_lstm_midi_model.py
   ```
3. Find generated music in `generated_music/`

## Technical Details
- Customizable sequence length and LSTM parameters
- Automatic instrument detection and mapping
- Grid search capability for hyperparameter optimization
- Scales data for improved model performance
