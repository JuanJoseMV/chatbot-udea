import re
import json
import pandas as pd

if __name__ == '__main__':
    
    with open('data/Reglamento Estudiantil_documento_34866874.txt') as f:
        data = f.read().splitlines()

    """
    The data variable contains a text divided by lines. The text is divided into "Artículos", 
    each one can be divided into either "literales" or "parágrafos". Loop through the lines. 
    Store the results in an reglamento dictionary. The keys of the dictionary are the 
    article's names, i.e. "ARTÍCULO [NUMBER]". The values of the dictionary should be either 
    "texto", "Parágrafo [NUMBER]", or "Literal [LETTER]". If a line starts with "ARTÍCULO [NUMBER]." 
    the text following is the text of the article and should be stored in the "texto" key. 
    If the line starts with "Parágrafo [NUMBER]", the text following is the text of the paragraph 
    and should be stored in the "Parágrafo [NUMBER]" key. If the line starts with a subdivision 
    (e.g.. "a.", "b.", "c.") that's called a "literal" and the text after it should be stored in 
    the "Literal [LETTER]" key. If the text of a line starts with "Concordancias:" that text should 
    be skipped and all the following lines after it until another article, paragraph, or literal is 
    found. The hierarchy has two higher levels: "TÍTULOS" and "CAPÍTULOS". "TÍTULOS" contain 
    "CAPÍTULOS" that contain "ARTÍCULOS" which contain "Parágrafos" or "Literales". So the code should 
    also check for "CAPÍTULOS" and "TÍTULOS". The highest level of the hierarchy should be the titles. 
    Every title should start with a chapter 0. If there is a chapter after chapter 0, all the following 
    articles should belong to this new chapter.
    """

    reglamento = {}
    current_title = None
    current_chapter = 'CAPÍTULO 0'
    current_article = None
    current_key = None
    skip_lines = False

    for line in data:
        line = line.strip()
        
        # Skip empty lines
        if not line:
            continue

        # Skip lines that are just numbers
        if line.isdigit():
            continue
        
        # Check for "TÍTULO [NAME]"
        title_match = re.match(r'TÍTULO (.+)', line)
        if title_match:
            skip_lines = False
            current_title = f'TÍTULO {title_match.group(1)}'
            reglamento[current_title] = {'CAPÍTULO 0': {}}
            current_chapter = 'CAPÍTULO 0'
            current_article = None
            current_key = None
            continue
        
        # Check for "CAPÍTULO [NAME]"
        chapter_match = re.match(r'CAPÍTULO (.+)', line)
        if chapter_match:
            skip_lines = False
            current_chapter = f'CAPÍTULO {chapter_match.group(1)}'
            reglamento[current_title][current_chapter] = {}
            current_article = None
            current_key = None
            continue
        
        # Check for "ARTÍCULO [NUMBER]"
        article_match = re.match(r'ARTÍCULO (\d+)\.', line)
        if article_match:
            skip_lines = False
            current_article = f'ARTÍCULO {article_match.group(1)}'
            reglamento[current_title][current_chapter][current_article] = {'texto': line[len(current_article) + 2:].strip()}
            current_key = 'texto'
            continue
        
        # Check for "Parágrafo [NUMBER]"
        paragrafo_match = re.match(r'Parágrafo (\d+)', line)
        if paragrafo_match:
            skip_lines = False
            current_key = f'Parágrafo {paragrafo_match.group(1)}'
            reglamento[current_title][current_chapter][current_article][current_key] = line[len(current_key) + 1:].strip()
            continue
        
        # Check for "Literal [LETTER]"
        literal_match = re.match(r'([a-z])\.', line)
        if literal_match:
            skip_lines = False
            current_key = f'Literal {literal_match.group(1)}'
            if current_key in reglamento[current_title][current_chapter][current_article]:
                current_key = f'Literal {literal_match.group(1)} (2)'
            reglamento[current_title][current_chapter][current_article][current_key] = line[len(literal_match.group(1)) + 1:].strip()
            continue
        
        # Check for "Concordancias:" or "Reglamentación:"
        if line.startswith('Concordancias:') or line.startswith('Reglamentación:') or line.startswith('Nota de editor:'):
            current_key = None
            skip_lines = True
            continue
        
        # Stop skipping lines when a new article, paragraph, or literal is found
        if skip_lines and (re.match(r'ARTÍCULO (\d+)\.', line) or re.match(r'Parágrafo (\d+)', line) or re.match(r'([a-z])\.', line)):
            skip_lines = False
        
        # Skip lines if skip_lines flag is set
        if skip_lines:
            continue
        
        # Append text to the current key if it exists
        if current_key and current_article and current_key in reglamento[current_title][current_chapter][current_article]:
            reglamento[current_title][current_chapter][current_article][current_key] += ' ' + line


# TODO: Save to a JSON file
    print(reglamento)

    with open('data/reglamento.json', 'w') as f:
        # save the dictionary to a json file, using utf-8 encoding
        json.dump(reglamento, f, ensure_ascii=False, indent=4)

