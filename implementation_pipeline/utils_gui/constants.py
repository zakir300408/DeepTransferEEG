show_rest_duration = 4_000      # ms
show_fixation_duration = 1_500  # ms
show_stimulus_duration = 6_000  # ms
TRIAL_DURATION = 4000  # ms, total duration of each trial
ArmMovementDuration = 8  # seconds, time for the exoskeleton to move arm down
DELAY_POST_STIMULUS = 500  # ms, delay after stimulus before segmentation


Cross_Symbol = "＋"       # Unicode for the fixation cross
Stimulus_Symbol = "⊞"     # Unicode for square box stimulus in green

#unicode for plus sign with a hollow

Cross_Size = 512   # Font size for fixation cross and stimulus
Stimulus_Symbol_Size = 512  # Font size for square box


### MODEL SPECIFIC CONSTANTS ###
SEEDS = [2, 3, 4, 5, 6, 7]  # Seeds for ensemble models
SAMPLE_RATE = 100  # Sample rate for EEG data in Hz
LR = 0.001
CHN = 27
TIME_SAMPLE_NUM = 725
FEATURE_DEEP_DIM = 704
MAX_TTA=25
STRIDE=1
STEPS=1
T=2
CONF_THRESH=0.2





#######READ LSL CONSTANTS#######

"""What: This module provides constants used in the EEG trial processing pipeline.
    These constants define the LSL stream name, sampling rates, trial timing,
    filter settings, spectrogram parameters, and desired EEG channels.
    """


########################################################################