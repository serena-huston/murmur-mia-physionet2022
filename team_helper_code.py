#!/usr/bin/env python

# Copied from helper_code, but with bugs ironed out.

import os, numpy as np, scipy as sp, scipy.io.wavfile

# Check if a variable is a number or represents a number.
def is_number(x):
    try:
        float(x)
        return True
    except (ValueError, TypeError):
        return False

# Check if a variable is an integer or represents an integer.
def is_integer(x):
    if is_number(x):
        return float(x).is_integer()
    else:
        return False

# Check if a variable is a a finite number or represents a finite number.
def is_finite_number(x):
    if is_number(x):
        return np.isfinite(float(x))
    else:
        return False

# Compare normalized strings.
def compare_strings(x, y):
    if x is not None and y is not None:
        try:
            return x.strip().casefold()==y.strip().casefold()
        except AttributeError: # For Python 2.x compatibility
            return x.strip().lower()==y.strip().lower()
    else:
        return False

# Find patient data files.
def find_patient_files(data_folder):
    # Find patient files.
    filenames = list()
    for f in sorted(os.listdir(data_folder)):
        root, extension = os.path.splitext(f)
        if not root.startswith('.') and extension=='.txt':
            filename = os.path.join(data_folder, f)
            filenames.append(filename)

    # To help with debugging, sort numerically if the filenames are integers.
    roots = [os.path.split(filename)[1][:-4] for filename in filenames]
    if all(is_integer(root) for root in roots):
        filenames = sorted(filenames, key=lambda filename: int(os.path.split(filename)[1][:-4]))

    return filenames


# Load patient data as a string.
def load_patient_data(filename):
    with open(filename, 'r') as f:
        data = f.read()
    return data


# Load a WAV file.
def load_wav_file(filename):
    frequency, recording = sp.io.wavfile.read(filename)
    return recording, frequency


# Load recordings.
def load_recordings(data_folder, data, get_frequencies=False):
    num_locations = get_num_locations(data)
    recording_information = data.split('\n')[1:num_locations+1]

    recordings = list()
    frequencies = list()
    for i in range(num_locations):
        entries = recording_information[i].split(' ')
        recording_file = entries[2]
        filename = os.path.join(data_folder, recording_file)
        recording, frequency = load_wav_file(filename)
        recordings.append(recording)
        frequencies.append(frequency)

    if get_frequencies:
        return recordings, frequencies
    else:
        return recordings


# Get patient ID from patient data.
def get_patient_id(data):
    patient_id = None
    for i, l in enumerate(data.split('\n')):
        if i==0:
            try:
                patient_id = l.split(' ')[0]
            except:
                pass
        else:
            break
    return patient_id


# Get number of recording locations from patient data.
def get_num_locations(data):
    num_locations = None
    for i, l in enumerate(data.split('\n')):
        if i==0:
            try:
                num_locations = int(l.split(' ')[1])
            except:
                pass
        else:
            break
    return num_locations


# Get recording locations from patient data.
def get_locations(data):
    num_locations = get_num_locations(data)
    locations = list()
    for i, l in enumerate(data.split('\n')):
        entries = l.split(' ')
        if i==0:
            pass
        elif 1<=i<=num_locations:
            locations.append(entries[0])
        else:
            break
    return locations


# flag locations from which recordings are available
def one_hot_encode_locations(data):
    oh_locs = np.zeros(5) # {'AV', 'MV', 'PV', 'TV', 'Phc'}
    locations = get_locations(data)
    if 'AV' in locations:
        oh_locs[0] = 1
    if 'MV' in locations:
        oh_locs[1] = 1
    if 'PV' in locations:
        oh_locs[2] = 1
    if 'TV' in locations:
        oh_locs[3] = 1
    if 'Phc' in locations:
        oh_locs[4] = 1
        
    return oh_locs


# Get frequency from patient data.
def get_frequency(data):
    frequency = None
    for i, l in enumerate(data.split('\n')):
        if i==0:
            try:
                frequency = float(l.split(' ')[1])
            except:
                pass
        else:
            break
    return frequency


# Get age from patient data.
def get_age(data):
    age = None
    for l in data.split('\n'):
        if l.startswith('#Age:'):
            try:
                age = l.split(': ')[1].strip()
            except:
                pass
    return age


# Get sex from patient data.
def get_sex(data):
    sex = None
    for l in data.split('\n'):
        if l.startswith('#Sex:'):
            try:
                sex = l.split(': ')[1].strip()
            except:
                pass
    return sex


# Get height from patient data.
def get_height(data):
    height = None
    for l in data.split('\n'):
        if l.startswith('#Height:'):
            try:
                height = float(l.split(': ')[1].strip())
            except:
                pass
    return height


# Get weight from patient data.
def get_weight(data):
    weight = None
    for l in data.split('\n'):
        if l.startswith('#Weight:'):
            try:
                weight = float(l.split(': ')[1].strip())
            except:
                pass
    return weight


# Get pregnancy status from patient data.
def get_pregnancy_status(data):
    is_pregnant = None
    for l in data.split('\n'):
        if l.startswith('#Pregnancy status:'):
            try:
                is_pregnant = bool(l.split(': ')[1].strip()=='True')
            except:
                pass
    return is_pregnant


# Get labels from patient data.
def get_label(data):
    label = None
    for l in data.split('\n'):
        if l.startswith('#Murmur:'):
            try:
                label = l.split(': ')[1]
            except:
                pass
    return label


# Get outcome from patient data.
def get_outcome(data):
    outcome = None
    for l in data.split('\n'):
        if l.startswith('#Outcome:'):
            try:
                outcome = l.split(': ')[1]
            except:
                pass
    if outcome is None:
        raise ValueError('No outcome available. Is your code trying to load labels from the hidden data?')
    return outcome


# Sanitize binary values from Challenge outputs.
def sanitize_binary_value(x):
    x = x.replace('"', '').replace("'", "").strip() # Remove any quotes or invisible characters.
    if (is_finite_number(x) and float(x)==1) or (x in ('True', 'true', 'T', 't')):
        return 1
    else:
        return 0


# Santize scalar values from Challenge outputs.
def sanitize_scalar_value(x):
    x = x.replace('"', '').replace("'", "").strip() # Remove any quotes or invisible characters.
    if is_finite_number(x) or (is_number(x) and (float(x)==float('inf') or float(x)==-float('inf'))):
        return float(x)
    else:
        return 0.0


# Save Challenge outputs.
def save_challenge_outputs(filename, patient_id, classes, labels, probabilities):
    # Format Challenge outputs.
    recording_string = '#{}'.format(patient_id)
    class_string = ','.join(str(c) for c in classes)
    label_string = ','.join(str(l) for l in labels)
    probabilities_string = ','.join(str(p) for p in probabilities)
    output_string = recording_string + '\n' + class_string + '\n' + label_string + '\n' + probabilities_string + '\n'

    # Write the Challenge outputs.
    with open(filename, 'w') as f:
        f.write(output_string)


# Load Challenge outputs.
def load_challenge_outputs(filename):
    with open(filename, 'r') as f:
        for i, l in enumerate(f):
            if i==0:
                patient_id = l[1:] if len(l)>1 else None
            elif i==1:
                classes = tuple(entry.strip() for entry in l.split(','))
            elif i==2:
                labels = tuple(sanitize_binary_value(entry) for entry in l.split(','))
            elif i==3:
                probabilities = tuple(sanitize_scalar_value(entry) for entry in l.split(','))
            else:
                break
    return patient_id, classes, labels, probabilities


# Get locations that murmur is audible.
def get_murmur_location(data):
    locs = None
    for l in data.split('\n'):
        if l.startswith('#Murmur locations:'):
            try:
                locs = l.split(': ')[1].strip()
                locs = locs.split('+')
            except:
                pass
    return locs


# Binary encoding. Returns 1 if label=='Present' or 'Unknown' else 0. 
def get_binary_labels(data):
    l = get_label(data) # (May be more robust to check if valid l?)
    label = int(l=='Present' or l=='Unknown') 
    return label


# Class target label: 0 if 'Present', 1 if 'Unknown', 2 if 'Absent'
def get_class_labels(data):
    label = None
    try:
        l = get_label(data)
    except ValueError:
        return 0
    
    if compare_strings(l,'Present'):
        label = 0
    elif compare_strings(l,'Unknown'):
        label = 1
    elif compare_strings(l, 'Absent'):
        label = 2

    return label 


# Outcome target label: 0 if 'Abnormal', 1 if 'Normal'
def get_outcome_labels(data):
    label = None
    try:
        l = get_outcome(data)
    except ValueError:
        return np.random.randint(2)

    if compare_strings(l, 'Abnormal'):
        label = 0
    elif compare_strings(l, 'Normal'):
        label = 1

    return label


# Returns a list of len(n_entries) of duplicate entries of x. 
def like_length(x, array_to_match):
    return [x] * array_to_match.shape[0]