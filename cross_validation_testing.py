import sys, os

sys.path.append(os.path.join(sys.path[0], '..'))

import pickle
import numpy as np
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
import team_code, team_constants, team_helper_code
from datetime import datetime
import evaluate_model, helper_code, combine_models


def load_pickled_data(fn):
    data = pickle.load(open(fn, 'rb'))

    patient_data = data['patient_data'] 
    patient_recordings = data['patient_recordings'] 
    recording_segmentations = data['recording_segmentations'] 
    try: 
        heart_rates = data['heart_rates']
    except KeyError:
        heart_rates = None

    return patient_data, patient_recordings, recording_segmentations, heart_rates


def eval_model_wrapper(classes, predictions, probabilities, murmur_labels, outcome_labels):
    # Define murmur and outcome classes.
    murmur_classes = ['Present', 'Unknown', 'Absent']
    outcome_classes = ['Abnormal', 'Normal']

    # Load and parse label and model outputs.
    murmur_binary_outputs = np.vstack(predictions)[:,:3] # take only first three entries per row for murmur classes
    murmur_scalar_outputs = np.vstack(probabilities)[:,:3]
    outcome_binary_outputs = np.vstack(predictions)[:,3:] # take final two for outcome classes
    outcome_scalar_outputs = np.vstack(probabilities)[:,3:]

    # For each patient, set the 'Present' or 'Abnormal' class to positive if no class is positive or if multiple classes are positive.
    murmur_labels = evaluate_model.enforce_positives(murmur_labels, murmur_classes, 'Present')
    murmur_binary_outputs = evaluate_model.enforce_positives(murmur_binary_outputs, murmur_classes, 'Present')
    outcome_labels = evaluate_model.enforce_positives(outcome_labels, outcome_classes, 'Abnormal')
    outcome_binary_outputs = evaluate_model.enforce_positives(outcome_binary_outputs, outcome_classes, 'Abnormal')

    # Evaluate the murmur model by comparing the labels and model outputs.
    murmur_auroc, murmur_auprc, murmur_auroc_classes, murmur_auprc_classes = evaluate_model.compute_auc(murmur_labels, murmur_scalar_outputs)
    murmur_f_measure, murmur_f_measure_classes = evaluate_model.compute_f_measure(murmur_labels, murmur_binary_outputs)
    murmur_accuracy, murmur_accuracy_classes = evaluate_model.compute_accuracy(murmur_labels, murmur_binary_outputs)
    murmur_weighted_accuracy = evaluate_model.compute_weighted_accuracy(murmur_labels, murmur_binary_outputs, murmur_classes) # This is the murmur scoring metric.
    murmur_cost = evaluate_model.compute_cost(outcome_labels, murmur_binary_outputs, outcome_classes, murmur_classes) # Use *outcomes* to score *murmurs* for the Challenge cost metric, but this is not the actual murmur scoring metric.
    murmur_scores = (murmur_classes, murmur_auroc, murmur_auprc, murmur_auroc_classes, murmur_auprc_classes, \
        murmur_f_measure, murmur_f_measure_classes, murmur_accuracy, murmur_accuracy_classes, murmur_weighted_accuracy, murmur_cost)

    # Evaluate the outcome model by comparing the labels and model outputs.
    outcome_auroc, outcome_auprc, outcome_auroc_classes, outcome_auprc_classes = evaluate_model.compute_auc(outcome_labels, outcome_scalar_outputs)
    outcome_f_measure, outcome_f_measure_classes = evaluate_model.compute_f_measure(outcome_labels, outcome_binary_outputs)
    outcome_accuracy, outcome_accuracy_classes = evaluate_model.compute_accuracy(outcome_labels, outcome_binary_outputs)
    outcome_weighted_accuracy = evaluate_model.compute_weighted_accuracy(outcome_labels, outcome_binary_outputs, outcome_classes)
    outcome_cost = evaluate_model.compute_cost(outcome_labels, outcome_binary_outputs, outcome_classes, outcome_classes) # This is the clinical outcomes scoring metric.
    outcome_scores = (outcome_classes, outcome_auroc, outcome_auprc, outcome_auroc_classes, outcome_auprc_classes, \
        outcome_f_measure, outcome_f_measure_classes, outcome_accuracy, outcome_accuracy_classes, outcome_weighted_accuracy, outcome_cost)

    # Return the results.
    classes, auroc, auprc, auroc_classes, auprc_classes, f_measure, f_measure_classes, accuracy, accuracy_classes, weighted_accuracy, cost = murmur_scores
    murmur_output_string = 'AUROC,AUPRC,F-measure,Accuracy,Weighted Accuracy,Cost\n{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}\n'.format(auroc, auprc, f_measure, accuracy, weighted_accuracy, cost)
    murmur_class_output_string = 'Classes,{}\nAUROC,{}\nAUPRC,{}\nF-measure,{}\nAccuracy,{}\n'.format(
        ','.join(classes),
        ','.join('{:.3f}'.format(x) for x in auroc_classes),
        ','.join('{:.3f}'.format(x) for x in auprc_classes),
        ','.join('{:.3f}'.format(x) for x in f_measure_classes),
        ','.join('{:.3f}'.format(x) for x in accuracy_classes))

    classes, auroc, auprc, auroc_classes, auprc_classes, f_measure, f_measure_classes, accuracy, accuracy_classes, weighted_accuracy, cost = outcome_scores
    outcome_output_string = 'AUROC,AUPRC,F-measure,Accuracy,Weighted Accuracy,Cost\n{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}\n'.format(auroc, auprc, f_measure, accuracy, weighted_accuracy, cost)
    outcome_class_output_string = 'Classes,{}\nAUROC,{}\nAUPRC,{}\nF-measure,{}\nAccuracy,{}\n'.format(
        ','.join(classes),
        ','.join('{:.3f}'.format(x) for x in auroc_classes),
        ','.join('{:.3f}'.format(x) for x in auprc_classes),
        ','.join('{:.3f}'.format(x) for x in f_measure_classes),
        ','.join('{:.3f}'.format(x) for x in accuracy_classes))

    output_string = '#Murmur scores\n' + murmur_output_string + '\n#Outcome scores\n' + outcome_output_string \
        + '\n#Murmur scores (per class)\n' + murmur_class_output_string + '\n#Outcome scores (per class)\n' + outcome_class_output_string

    print(output_string)

    return output_string


def cv_run_model(patient_data, patient_recordings, tsv_annotations, verbose, train=True):

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    labels = [] # needed if using skf
    for p in range(len(patient_data)):
        labels.append(team_helper_code.get_class_labels(patient_data[p]))

    results = ''
    split_outputs = {}

    for split, (train_index, test_index) in enumerate(kf.split(patient_data, labels)):
        print(f'############ SPLIT {split+1} ##############')
        results += f'\n############ SPLIT {split+1} ##############\n'
        model_folder = f'models/split{split+1}'
        team_model = {}
      
        # train model
        if train:
            team_code.train_model(patient_data[train_index], 
                                patient_recordings[train_index], 
                                tsv_annotations[train_index], 
                                model_folder, 
                                verbose, 
                                given_segmentations=None,
                                given_hrs=None)

        # load model
        team_model = team_code.load_challenge_model(model_folder, verbose)

        # run and test model
        classes = [] # concatenated murmur_classes + outcome_classes, just the strings 
        predictions = [] # concatenated murmur_pred + outcome_pred
        probabilities = [] # concatenated murmur_probs + outcome_probs

        n_test = len(patient_data[test_index])
        murmur_targets = np.zeros((n_test, 3), dtype=np.bool)
        outcome_targets = np.zeros((n_test, 2), dtype=np.bool)

        # for individual recording results, to test thresholds
        murmur_target_per_recording = []
        murmur_probs_gb = []
        murmur_probs_cnn = []
        murmur_probs_gb = []
        outlier_preds_murmur = []

        outcome_target_per_recording = []
        outcome_probs_gb = []
        outlier_preds_outcome = []
        
        for i, (data, recordings) in enumerate(zip(patient_data[test_index], patient_recordings[test_index])):
            murmur_classes = ['Present', 'Unknown', 'Absent']
            outcome_classes = ['Abnormal', 'Normal']

            ########## UNCOMMENT BELOW FOR PER-RECORDING EVAL ###########
            outlier_probs_murmur, gb_probs_murmur, cnn_probs_murmur, \
                outlier_probs_outcome, gb_probs_outcome =\
                    team_code.run_model(team_model, data, recordings, verbose,
                                        given_segmentations=None, given_hrs=None)
            
            # cl, preds, probs = team_code.run_challenge_model(team_model, data, recordings, verbose)

            # combine cnn_probs_outcome with gb_probs_outcome using logistic regression
            pred_labels_murmur, all_probs_murmur = combine_models.combine_CNN_GB(cnn_probs_murmur, gb_probs_murmur)

            # Combine predictions from different models
            murmur_pred, murmur_probs = \
                combine_models.recording_to_murmur_predictions(outlier_probs_murmur, all_probs_murmur)
            outcome_pred, outcome_probs = \
                combine_models.recording_to_outcome_predictions(outlier_probs_outcome, gb_probs_outcome)
            
            # Concatenate classes, labels, and probabilities.
            cl = murmur_classes + outcome_classes
            preds = np.concatenate((murmur_pred, outcome_pred))
            probs = np.concatenate((murmur_probs, outcome_probs))

            ########### FOR WHOLE RUN_MODEL CV #############
            # cl, preds, probs = team_code.run_challenge_model(team_model, data, recordings, verbose)

            classes.append(cl)
            predictions.append(preds)
            probabilities.append(probs)

            murmur_label = helper_code.get_murmur(data)
            for j, x in enumerate(cl[:3]):
                if helper_code.compare_strings(murmur_label, x):
                    murmur_targets[i, j] = 1

            outcome_label = helper_code.get_outcome(data)
            for j, x in enumerate(cl[3:]):
                if helper_code.compare_strings(outcome_label, x):
                    outcome_targets[i, j] = 1

            # for testing only
            murmur_target_per_recording.append(murmur_targets[i])
            murmur_probs_cnn.append(cnn_probs_murmur)
            murmur_probs_gb.append(gb_probs_murmur)
            outlier_preds_murmur.append(outlier_probs_murmur)

            outcome_target_per_recording.append(outcome_targets[i])
            outcome_probs_gb.append(gb_probs_outcome)
            outlier_preds_outcome.append(outlier_probs_outcome)

        # save outputs to dict for testing
        split_outputs[f'split {split} murmur targets'] = murmur_target_per_recording
        split_outputs[f'split {split} gb murmur probabilities'] = murmur_probs_gb
        split_outputs[f'split {split} cnn murmur probabilities'] = murmur_probs_cnn
        split_outputs[f'split {split} murmur outlier probabilities'] = outlier_preds_murmur
        
        split_outputs[f'split {split} outcome targets'] = outcome_target_per_recording
        split_outputs[f'split {split} gb outcome probabilities'] = outcome_probs_gb
        split_outputs[f'split {split} outcome outlier probabilities'] = outlier_preds_outcome

        output_str = eval_model_wrapper(classes, predictions, probabilities, murmur_targets, outcome_targets)  

        results += output_str

    return results, split_outputs

def main(data_folder, verbose):
    patient_data_l, patient_recordings_l, tsv_annotations_l = \
        team_code.load_data_from_folder(data_folder, load_segmentations=True)

    patient_data = np.array(patient_data_l)
    patient_recordings = np.array(patient_recordings_l, dtype=object)
    tsv_annotations = np.array(tsv_annotations_l, dtype=object)
    
    results, split_outputs = cv_run_model(patient_data, patient_recordings, tsv_annotations, 
                            verbose,
                             train=True)

    s = 'run-model-' + datetime.now().strftime("%Y%m%d-%H%M")
    folder = 'cnn-results'
    os.makedirs(folder, exist_ok=True) 
    fn = os.path.join(folder, s + '.txt')
    with open(fn, 'w') as f:
        f.write(results)

    # pickle outputs as well
    pn = os.path.join(folder, s + '.sav')
    pickle.dump(split_outputs, open(pn, 'wb'))


DATA_PATH = "/Users/serenahuston/GitRepos/Data/DataSubset_12_Patients"
main(DATA_PATH, 2)