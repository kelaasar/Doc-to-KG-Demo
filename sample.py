import os
from pdf2docx import Converter
from docx import Document
from pptx import Presentation
from docx.table import Table
from bs4 import BeautifulSoup
from html import escape


# convert to html

def pdf_to_docx(pdf_path, output_dir):
    base_name = os.path.basename(pdf_path).replace('.pdf', '')
    docx_path = os.path.join(output_dir, base_name + '.docx')

    # Convert PDF to DOCX
    cv = Converter(pdf_path)
    cv.convert(docx_path, start=0, end=None)
    cv.close()

    # Load the DOCX and remove footnotes and headnotes
    doc = Document(docx_path)
    clean_doc = remove_footnotes_and_headnotes(doc)
    clean_doc.save(docx_path)  # Overwrite the original DOCX file

    return docx_path

def remove_footnotes_and_headnotes(doc):
    new_doc = Document()

    # Copy all the paragraphs and tables from the original document
    for element in doc.element.body:
        new_doc.element.body.append(element)

    for section in new_doc.sections:
        # Remove header content
        for paragraph in section.header.paragraphs:
            p = paragraph._element
            p.getparent().remove(p)

        # Remove footer content
        for paragraph in section.footer.paragraphs:
            p = paragraph._element
            p.getparent().remove(p)

    return new_doc

def docx_to_html(docx_path):
    doc = Document(docx_path)
    html = '<html><body>'

    def extract_text_from_cell(cell):
        paragraphs = cell.paragraphs
        return ''.join(escape(p.text) for p in paragraphs)

    def table_to_html(table):
        table_html = '<table>'
        for row in table.rows:
            table_html += '<tr>'
            for cell in row.cells:
                cell_text = extract_text_from_cell(cell)
                table_html += f'<td>{cell_text}</td>'
            table_html += '</tr>'
        table_html += '</table>'
        return table_html

    for element in doc.element.body:
        if element.tag.endswith('p'):
            para = element
            html += f'<p>{escape(para.text)}</p>'
        elif element.tag.endswith('tbl'):
            table = Table(element, doc)
            table_html = table_to_html(table)
            json_obj = table_to_json(BeautifulSoup(table_html, 'lxml').find('table'))
            html += f'<pre>{json_obj}</pre>'

    html += '</body></html>'
    return html

def pptx_to_html(pptx_path):
    prs = Presentation(pptx_path)
    html = '<html><body>'

    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, 'text'):
                html += f'<p>{escape(shape.text)}</p>'

    html += '</body></html>'
    return html

def file_to_html(file_path, html_path):
    if file_path.endswith('.pdf'):
        output_dir = os.path.dirname(html_path)
        # Convert PDF to DOCX
        docx_path = pdf_to_docx(file_path, output_dir)
        print(f'PDF converted to DOCX: {docx_path}')
        html = docx_to_html(docx_path)
    elif file_path.endswith('.docx'):
        html = docx_to_html(file_path)
    elif file_path.endswith('.pptx'):
        html = pptx_to_html(file_path)
    else:
        raise ValueError("Unsupported file type. Please provide a .pdf, .docx, or .pptx file.")

    # Write HTML to file
    with open(html_path, 'w', encoding='utf-8') as html_file:
        html_file.write(html)
    return html

# Example usage
file_to_html('data/N_PR_8715_0026.pdf', 'data/converted/converted.html')
#file_to_html('data/N_PR_8715_0026.docx', 'data/converted/DOCX-converted.html')
# file_to_html('data/Chocolate Cake Recipe.pptx', 'data/converted/PPTX-converted.html')



# Create chunks grouped by similarity
from unstructured.partition.text import partition_text
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

file_path = "data/converted/converted.txt"
max_chars_per_chunk = 5000  # Set a hard-coded max character limit for each chunk
num_clusters = 20  # Desired number of clusters

paragraphs = partition_text(filename=file_path)
paragraph_texts = [p.text for p in paragraphs]
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
embeddings = model.encode(paragraph_texts)

def merge_clusters(cluster_a, cluster_b):
    merged_paragraphs = cluster_a['paragraphs'] + cluster_b['paragraphs']
    merged_embeddings = cluster_a['embeddings'] + cluster_b['embeddings']
    return {'paragraphs': merged_paragraphs, 'embeddings': merged_embeddings}

def get_cluster_character_count(cluster):
    return sum(len(p) for p in cluster['paragraphs'])

# Initialize each paragraph as its own cluster
grouped_paragraphs = [{'paragraphs': [paragraph], 'embeddings': [embedding]} for paragraph, embedding in zip(paragraph_texts, embeddings)]
current_num_clusters = len(grouped_paragraphs)

while current_num_clusters > num_clusters:
    best_similarity = -1
    best_pair = None
    
    # Calculate similarity between adjacent clusters
    for i in range(current_num_clusters - 1):
        cluster_a = grouped_paragraphs[i]
        cluster_b = grouped_paragraphs[i + 1]
        
        cluster_a_embedding = np.mean(cluster_a['embeddings'], axis=0)
        cluster_b_embedding = np.mean(cluster_b['embeddings'], axis=0)
        
        similarity = cosine_similarity([cluster_a_embedding], [cluster_b_embedding])[0][0]
        
        combined_char_count = get_cluster_character_count(cluster_a) + get_cluster_character_count(cluster_b)
        
        # Only consider merging if the character count is within the limit
        if similarity > best_similarity and combined_char_count <= max_chars_per_chunk:
            best_similarity = similarity
            best_pair = (i, i + 1)
    
    # If no valid pair to merge was found, break the loop
    if best_pair is None:
        break
    
    # Merge the most similar adjacent clusters that respect the character limit
    cluster_a_index, cluster_b_index = best_pair
    merged_cluster = merge_clusters(grouped_paragraphs[cluster_a_index], grouped_paragraphs[cluster_b_index])
    
    # Replace the first cluster with the merged one and remove the second
    grouped_paragraphs[cluster_a_index] = merged_cluster
    del grouped_paragraphs[cluster_b_index]
    
    # Update the current number of clusters
    current_num_clusters -= 1

# Print clusters
for cluster_idx, cluster in enumerate(grouped_paragraphs):
    print(f"\nCluster {cluster_idx}:")
    for paragraph in cluster['paragraphs']:
        print(paragraph)


# Add descriptive labels

import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import json

# load the .env file
_ = load_dotenv(find_dotenv())
client = OpenAI(
   api_key=os.environ.get("OPENAI_API_KEY")
)

model = "gpt-4o"
temperature = 0

chunks = ['\n'.join(cluster['paragraphs']) for cluster in grouped_paragraphs]

def get_summary(client, model, messages, temperature):
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return completion.choices[0].message.content

chunksDict = {}

for chunk in chunks:
    prompt = f"""#
    For the given chunk, I want you to return a dictionary of two elements, the key being a short descriptive label of the chunk, and the value being the chunk content. 
    Below is the output schema:
    {{
        "description": "short desc", 
        "content": "long string of document chunk content"
    }},
    """
    
    messages = [
        {"role": "system", "content": "You are an AI that generates descriptive labels for text chunks."},
        {"role": "user", "content": chunk},
        {"role": "user", "content": prompt},
    ]
    
    response = get_summary(client, model, messages, temperature)
    
    response_json = response.strip().replace("```json", "").replace("```", "")
    
    try:
        response_dict = json.loads(response_json)  
    except json.JSONDecodeError as e:
        print("Failed to parse response as JSON:", e)
        continue
    
    chunksDict[response_dict["description"]] = response_dict["content"]
