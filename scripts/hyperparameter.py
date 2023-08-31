import re
import os
import pandas as pd

def extract_values(filename, patterns):
    with open(filename, 'r') as file:
        content = file.read()

    extracted_values = {}
    
    # Extract model name
    regex_name = r'"model_name"\s*:\s*"([^"]+)"'
    match_name = re.search(regex_name, content)
    if match_name:
        extracted_values['model_name'] = match_name.group(1)
    
    # Extract F1 score
    regex_f1 = r"F1\s*=\s*([\d\.]+)"
    match_f1 = re.findall(regex_f1, content)
    if match_f1:
        extracted_values['F1'] = match_f1[4]

    # Extract other patterns
    for pattern in patterns:
        regex = fr"--{pattern}\s+(\d+(\.\d+)?)"
        match = re.search(regex, content)
        if match:
            extracted_values[pattern] = float(match.group(1))
        else:
            extracted_values[pattern] = None  # set to None if the pattern is not found
            
    return extracted_values

if __name__ == '__main__':
    directory = '/root/neuralcodesum/tmp/'
    patterns = ['learning_rate', 'batch_size', 'nlayers', 'd_ff']
    
    # Filter for files that match the desired pattern
    files = [f for f in os.listdir(directory) if re.match(r'model_\d+\.txt', f)]

    # Initialize empty list to hold data for DataFrame
    data = []
    
    # Extract values for each file and append to the data list
    for file in files:
        filepath = os.path.join(directory, file)
        data.append(extract_values(filepath, patterns))
    
    # Convert list of dictionaries to DataFrame
    df = pd.DataFrame(data)
    df = df.sort_values(by='F1', ascending=False)
    df.to_html('test_2.html')
    print(df)
