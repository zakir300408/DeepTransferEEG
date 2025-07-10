show_rest_duration = 4_000      # ms
show_fixation_duration = 1_500  # ms
show_stimulus_duration = 6_000  # ms
TRIAL_DURATION = 4000  # ms, total duration of each trial
ArmMovementDuration = 10  # seconds, time for the exoskeleton to move arm down
DELAY_POST_STIMULUS = 500  # ms, delay after stimulus before segmentation


Cross_Symbol = "准备开始"       # Unicode for the fixation cross
Stimulus_Symbol = "⊞"     # Unicode for square box stimulus in green
No_Stimulus_Symbol = "别动"
Prediction_Text = "(休息)"

STIMULUS_GIF_PATH = r"implementation_pipeline\Animation.gif"

#unicode for plus sign with a hollow

Cross_Size = 150   # Font size for fixation cross and stimulus
Text_Symbol_Size = 150  # Font size for text symbols
Stimulus_Symbol_Size = 150  # Font size for square box


### MODEL SPECIFIC CONSTANTS ###
### MODEL SPECIFIC CONSTANTS ###
SEEDS = [2, 4, 5]  # Seeds for ensemble models
SAMPLE_RATE = 100  # Sample rate for EEG data in Hz
LR = 0.001
CHN = 27
TIME_SAMPLE_NUM = 725
FEATURE_DEEP_DIM = 704
MAX_TTA=8  # was 10, reduced to trigger adaptation every trial for debug
STRIDE=1
STEPS=3
T=2
CONF_THRESH=0.5






#######READ LSL CONSTANTS#######

"""What: This module provides constants used in the EEG trial processing pipeline.
    These constants define the LSL stream name, sampling rates, trial timing,
    filter settings, spectrogram parameters, and desired EEG channels.
    """


########################################################################