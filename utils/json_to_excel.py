import json
import pandas as pd

# Step 1: Read the JSON file
with open('data/articles_expansion.json', 'r') as file:
    data = json.load(file)

# Step 2: Extract the articles and their corresponding questions
rows = []
for article, content in data.items():
    row = [article] + list(content['preguntas'].keys())
    rows.append(row)

# Step 3: Create a DataFrame with the appropriate columns
max_questions = max(len(row) - 1 for row in rows)
columns = ['artículo'] + [f'pregunta {i+1}' for i in range(max_questions)]
df = pd.DataFrame(rows, columns=columns)

# Step 4: Save the DataFrame as a CSV file
df.to_csv('data/articles.csv', index=False)