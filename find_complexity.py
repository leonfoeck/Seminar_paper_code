import json
import pandas as pd
import matplotlib.pyplot as plt

# Ihre Codezusammenführung
CodeComplex = []

with open("bigDataSet_10Epochs_beam_Top_1.json", 'r') as file:
    for line in file:
        data = json.loads(line)
        predictions = data['predictions'][0]
        references = data['references'][0]
        pred = False

        if references == predictions:
            pred = True

        CodeComplex.append([len(data['code'].split()), pred])

df = pd.DataFrame(CodeComplex, columns=['Token_Length', 'Prediction_result'])

# Definieren Sie die Bucket-Grenzen
bucket_limits = [0, 100, 200, 500, 1000, 2000, float('inf')]
bucket_names = ["[0, 100]", "(100, 200]", "(200, 500]", "(500, 1000]", "(1000, 2000]", "(2000, +∞)"]

# Erstellen Sie eine neue Spalte für den Bucket, zu dem jede Zeile gehört
df['Bucket'] = pd.cut(df['Token_Length'], bins=bucket_limits, labels=bucket_names, right=False)

# Berechnen Sie die Anzahl der Datenpunkte in jedem Bucket
bucket_counts = df['Bucket'].value_counts().sort_index()

# Berechnen Sie die Accuracy für jeden Bucket
bucket_accuracy = df.groupby('Bucket').Prediction_result.mean()

print(bucket_accuracy)

# Plotten Sie die Ergebnisse
ax = bucket_accuracy.plot(kind='bar', figsize=(10, 6))
plt.ylabel('Accuracy')
plt.xlabel('Token Length Buckets')
plt.title('Accuracy based on Code Complexity (Token Length)')

# Fügen Sie die Anzahl der Datenpunkte über jeden Balken hinzu
for i, v in enumerate(bucket_accuracy):
    ax.text(i, v + 0.01, str(bucket_counts[i]), ha='center', va='bottom', fontsize=9)

plt.show()
