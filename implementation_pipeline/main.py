import sys
import os
import numpy as np

from step_1_load_data import load_custom_epoch_data
from step_2_preprocess import preprocess_trial
from step_3_load_setup_model import setup_inference_pipeline, infer_pre_tta, infer_tta
from ttime_ensemble import SML  # add ensemble helper

# pipeline constants
DATASET_NAME = "CustomEpoch"
SAMPLE_RATE  = 100

def main_pipeline():
    print("Starting implementation pipeline...")

    # Step 1: Load and verify data
    X, labels, meta = load_custom_epoch_data(dataset_name=DATASET_NAME)
    print("Step 1: Data loaded successfully.")
    print("X shape:", X.shape)
    print("Labels shape:", labels.shape)
    print("Meta shape:", meta.shape)

    # Step 2: Preprocess Multiple Trials (at least 8 for TTA)
    print("\n--- Step 2: Preprocess Multiple Trials ---")
    num_trials = min(60, len(X))  # Test with 8 trials (minimum 8)
    print(f"Processing first {num_trials} trials...")
    
    processed_trials = []
    for i in range(num_trials):
        trial = X[i]  # shape (channels, time_samples)
        proc_trial = preprocess_trial(trial, SAMPLE_RATE)
        processed_trials.append(proc_trial)
        if i == 0:
            print(f"Raw trial shape: {trial.shape}")
            print(f"Preprocessed trial shape: {proc_trial.shape}")
    
    print(f"Preprocessed {len(processed_trials)} trials")

    # Step 3: Load Trained Model and Setup Inference
    print("\n--- Step 3: Load Trained Model ---")
    try:
        # Setup Pre-TTA pipeline
        model_pre, R_pre, args, mode = setup_inference_pipeline(
            data_name=DATASET_NAME, 
            subject_id=0, 
            seed=2, 
            mode='pre_tta'
        )
        print(f"Successfully loaded {mode} model")
        
        # Setup TTA pipeline
        (model_tta, optimizer, R_tta, data_cum), args_tta, _ = setup_inference_pipeline(
            data_name=DATASET_NAME,
            subject_id=0,
            seed=2,
            mode='tta'
        )
        print("Successfully loaded TTA model")
        
        # Step 4: Sequential Prediction on Multiple Trials
        print(f"\n--- Step 4: Sequential Prediction on {num_trials} Trials ---")
        
        # Storage for results
        pre_tta_predictions = []
        tta_predictions = []
        true_labels = labels[:num_trials]
        
        print("Trial | True | Pre-TTA | TTA   | Pre-Conf | TTA-Conf | TTA-Adapted")
        print("-" * 70)
        
        for trial_idx in range(num_trials):
            # Extract temporal part for model input
            proc_trial = processed_trials[trial_idx]
            trial_temporal = proc_trial[:, :args.time_sample_num]  # (chn, time_sample_num)
            
            # Pre-TTA prediction (independent for each trial)
            pred_probs_pre, R_pre = infer_pre_tta(
                model=model_pre,
                trial=trial_temporal,
                args=args,
                R=R_pre,
                trial_idx=trial_idx
            )
            
            # TTA prediction (sequential with adaptation)
            pred_probs_tta, R_tta, data_cum = infer_tta(
                model=model_tta,
                optimizer=optimizer,
                trial=trial_temporal,
                args=args_tta,
                R=R_tta,
                data_cum=data_cum,
                trial_idx=trial_idx
            )
            
            # Store predictions
            pre_tta_predictions.append(pred_probs_pre[0])
            tta_predictions.append(pred_probs_tta[0])
            
            # Get predicted classes and confidences
            pre_class = pred_probs_pre.argmax(axis=1)[0]
            tta_class = pred_probs_tta.argmax(axis=1)[0]
            pre_conf = pred_probs_pre.max()
            tta_conf = pred_probs_tta.max()
            
            # Check if TTA adaptation occurred (only after trial 7)
            adapted = "Yes" if trial_idx >= 7 else "No"
            
            print(f"{trial_idx:5d} | {true_labels[trial_idx]:4d} | {pre_class:7d} | {tta_class:5d} | "
                  f"{pre_conf:8.3f} | {tta_conf:8.3f} | {adapted:11s}")
        
        # Step 5: Comprehensive Evaluation
        print(f"\n--- Step 5: Evaluation Summary ---")
        
        # Convert to numpy arrays
        pre_tta_preds = np.array(pre_tta_predictions)
        tta_preds = np.array(tta_predictions)
        
        # Calculate accuracies for different trial ranges
        def calc_accuracy(predictions, labels):
            pred_classes = predictions.argmax(axis=1)
            return (pred_classes == labels).mean() * 100
        
        # Full sequence accuracy
        pre_acc_full = calc_accuracy(pre_tta_preds, true_labels)
        tta_acc_full = calc_accuracy(tta_preds, true_labels)
        
        print(f"Full sequence ({num_trials} trials):")
        print(f"  Pre-TTA accuracy: {pre_acc_full:.1f}%")
        print(f"  TTA accuracy: {tta_acc_full:.1f}%")
        print(f"  TTA improvement: {tta_acc_full - pre_acc_full:+.1f}%")
        
        # Before adaptation (trials 0-7)
        if num_trials > 8:
            pre_acc_early = calc_accuracy(pre_tta_preds[:8], true_labels[:8])
            tta_acc_early = calc_accuracy(tta_preds[:8], true_labels[:8])
            
            print(f"\nBefore adaptation (trials 0-7):")
            print(f"  Pre-TTA accuracy: {pre_acc_early:.1f}%")
            print(f"  TTA accuracy: {tta_acc_early:.1f}%")
            
            # After adaptation starts (trials 8+)
            pre_acc_late = calc_accuracy(pre_tta_preds[8:], true_labels[8:])
            tta_acc_late = calc_accuracy(tta_preds[8:], true_labels[8:])
            
            print(f"\nAfter adaptation starts (trials 8+):")
            print(f"  Pre-TTA accuracy: {pre_acc_late:.1f}%")
            print(f"  TTA accuracy: {tta_acc_late:.1f}%")
            print(f"  TTA improvement: {tta_acc_late - pre_acc_late:+.1f}%")
        
        # Confidence analysis
        pre_confs = pre_tta_preds.max(axis=1)
        tta_confs = tta_preds.max(axis=1)
        
        print(f"\nConfidence Analysis:")
        print(f"  Pre-TTA avg confidence: {pre_confs.mean():.3f} ± {pre_confs.std():.3f}")
        print(f"  TTA avg confidence: {tta_confs.mean():.3f} ± {tta_confs.std():.3f}")
        
        # Check adaptation trigger
        print(f"\nTTA Adaptation Info:")
        print(f"  Minimum trials for adaptation: {args_tta.max_tta}")
        print(f"  Confidence threshold: {args_tta.conf_thresh}")
        print(f"  Adaptation stride: {args_tta.stride}")
        print(f"  Number of adaptation steps: {args_tta.steps}")
        
        # Count high-confidence trials that could trigger adaptation
        high_conf_trials = (tta_confs >= args_tta.conf_thresh).sum()
        print(f"  Trials with confidence >= threshold: {high_conf_trials}/{num_trials}")
        
        adaptation_opportunities = sum(1 for i in range(args_tta.max_tta, num_trials, args_tta.stride) 
                                     if tta_confs[i] >= args_tta.conf_thresh)
        print(f"  Actual adaptation opportunities: {adaptation_opportunities}")
        
        # Step 6: Ensemble Across Seeds
        print("\n--- Step 6: Ensemble Across Seeds ---")
        SEEDS = [2,3,5,6,7,8,9,12]
        all_pre, all_tta = [], []
        for s in SEEDS:
            # setup pipelines for seed s
            model_pre_s, R_pre_s, args_s, _ = setup_inference_pipeline(DATASET_NAME, 0, s, 'pre_tta')
            (model_tta_s, opt_s, R_tta_s, data_cum_s), args_tta_s, _ = setup_inference_pipeline(DATASET_NAME, 0, s, 'tta')
            pre_s, tta_s = [], []
            for i in range(num_trials):
                p_pre, R_pre_s = infer_pre_tta(
                    model=model_pre_s,
                    trial=processed_trials[i][:, :args_s.time_sample_num],
                    args=args_s,
                    R=R_pre_s,
                    trial_idx=i
                )
                p_tta, R_tta_s, data_cum_s = infer_tta(
                    model=model_tta_s,
                    optimizer=opt_s,
                    trial=processed_trials[i][:, :args_tta_s.time_sample_num],
                    args=args_tta_s,
                    R=R_tta_s,
                    data_cum=data_cum_s,
                    trial_idx=i
                )
                # collect class-1 probabilities
                pre_s.append(p_pre[0,1])
                tta_s.append(p_tta[0,1])
            all_pre.append(pre_s)
            all_tta.append(tta_s)

        pre_arr = np.stack(all_pre)   # shape (n_seeds, num_trials)
        tta_arr = np.stack(all_tta)

        # ensemble votes
        mean_pre = (pre_arr.mean(0)   > 0.5).astype(int)
        med_pre  = (np.median(pre_arr,0)> 0.5).astype(int)
        sml_pre  = SML(pre_arr)

        mean_tta = (tta_arr.mean(0)   > 0.5).astype(int)
        med_tta  = (np.median(tta_arr,0)> 0.5).astype(int)
        sml_tta  = SML(tta_arr)

        acc = lambda pred: (pred == true_labels).mean() * 100
        print(f"Ensemble Pre-TTA Acc: Mean={acc(mean_pre):.1f}%, Median={acc(med_pre):.1f}%, SML={acc(sml_pre):.1f}%")
        print(f"Ensemble    TTA Acc: Mean={acc(mean_tta):.1f}%, Median={acc(med_tta):.1f}%, SML={acc(sml_tta):.1f}%")

        # Per-trial ensemble predictions
        print(f"\n--- Step 7: Ensemble Per-Trial Predictions on {num_trials} Trials ---")
        print("Trial | True | Pre-Mean | Pre-Median | Pre-SML | TTA-Mean | TTA-Median | TTA-SML")
        print("-" * 85)
        for trial_idx in range(num_trials):
            true_lbl = true_labels[trial_idx]
            pm = mean_pre[trial_idx]
            pmed = med_pre[trial_idx]
            psml = sml_pre[trial_idx]
            tm = mean_tta[trial_idx]
            tmed = med_tta[trial_idx]
            tsml = sml_tta[trial_idx]
            print(f"{trial_idx:5d} | {true_lbl:4d} | {pm:8d} | {pmed:10d} | {psml:7d} | "
                  f"{tm:9d} | {tmed:11d} | {tsml:6d}")

    except Exception as e:
        print(f"Model loading/prediction failed: {e}")
        import traceback
        traceback.print_exc()
        print("Make sure you have trained models in the ./runs/ directory")
    
    print("\nImplementation pipeline finished.")

if __name__ == "__main__":
    main_pipeline()
