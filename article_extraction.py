import re
import json
import pandas as pd

if __name__ == '__main__':
    
    with open('data/Reglamento Estudiantil_documento_34866874.txt') as f:
        data = f.read().splitlines()

    """
    The variable data contains a document divided in lines. Extract and join the line of text in 
    between the lines that start with "ARTÍCULO [NUMBER]", also include the text after "ARTÍCULO [NUMBER]". 
    save all the results in the articulos variable. When the line starts with either "Concordancias:",
    "Reglamentación:" or "Nota del editor." Skip the lines until another article is found.
    """

    articulos = {}
    skip_lines = False
    current_article = None
    for line in data:
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue

        # Skip lines that are just numbers
        if line.isdigit():
            continue
        
        # Check for "ARTÍCULO [NUMBER]"
        article_match = re.match(r'ARTÍCULO (\d+)\.', line)
        if article_match:
            current_article = article_match.group(0)
            articulos[current_article] = line[len(current_article):].strip()
            skip_lines = False
        elif current_article and any(line.startswith(prefix) for prefix in ["Concordancias:", "Reglamentación:", "Nota del editor."]):
            skip_lines = True
        elif current_article and not skip_lines:
            articulos[current_article] += ' ' + line

    with open('data/articles.json', 'w') as f:
        # save the dictionary to a json file, using utf-8 encoding
        json.dump(articulos, f, ensure_ascii=False, indent=4)

