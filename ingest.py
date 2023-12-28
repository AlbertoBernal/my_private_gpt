#!/usr/bin/env python3
import os
import glob
from typing import List, Callable
from dotenv import load_dotenv
from multiprocessing import Pool
from tqdm import tqdm
import pickle
from langchain.llms import GPT4All, LlamaCpp
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains.summarize import load_summarize_chain
from langchain import PromptTemplate
import numpy as np
from sklearn.cluster import KMeans
import pdb

from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.text_splitter            import Language

from langchain.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PDFMinerLoader,
    TextLoader,
    UnstructuredEPubLoader,
    UnstructuredEmailLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredRSTLoader,
    UnstructuredWordDocumentLoader,
)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings
from langchain.docstore.document import Document
from constants import CHROMA_SETTINGS

import argparse

load_dotenv()


#Load environment variables
persist_directory = os.environ.get('PERSIST_DIRECTORY')
source_directory = os.environ.get('SOURCE_DIRECTORY', 'source_documents')
embeddings_model_name = os.environ.get('EMBEDDINGS_MODEL_NAME')
DEFAULT_CHUNK_SIZE=4000
DEFAULT_CHUNK_OVERLAP=int(DEFAULT_CHUNK_SIZE*0.3)
model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX_SUMMARY')

def get_words(documents : List[Document]) -> List[int]:
    """ Function to retrieve max words found in a single document and the amount of total words
    Args:
        documents (List[Document]): 

    Returns:
        List[int,int]: max number of words of all documents, accumulated amount of tokens
    """
    text = ''
    saved_max = 0
    for doc in documents:
        text += doc.page_content
        saved_max =max(saved_max , len(text))
    return saved_max, len(text)

def get_perfect_chunk_size(source_dir : str, number_of_pieces : int = 0):
    #get list of documents
    documents: List[Document] = load_documents(source_dir)
    #load all documents and get the amount of tokens
    max_token_in_single_doc, tokens_amount = get_words(documents)
    mean_tokens_per_doc = tokens_amount / len(documents)
    # we could have big documents with chapters inside those documents
    # or we could have several documents , each one being a single chapter
    if number_of_pieces == 0:
        return min(max_token_in_single_doc, 6000)
    else: #let's say we want to split a document in 100 pieces, numbr of pieces should be 100 and we'll get how many token would have every chunk
        return tokens_amount/number_of_pieces
    
# Custom document loaders
class MyElmLoader(UnstructuredEmailLoader):
    """Wrapper to fallback to text/plain when default does not work"""

    def load(self) -> List[Document]:
        """Wrapper adding fallback for elm without html"""
        try:
            try:
                doc = UnstructuredEmailLoader.load(self)
            except ValueError as e:
                if 'text/html content not found in email' in str(e):
                    # Try plain text
                    self.unstructured_kwargs["content_source"]="text/plain"
                    doc = UnstructuredEmailLoader.load(self)
                else:
                    raise
        except Exception as e:
            # Add file_path to exception message
            raise type(e)(f"{self.file_path}: {e}") from e

        return doc


# Map file extensions to document loaders and their arguments
LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".eml": (MyElmLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PDFMinerLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    ".py":  (TextLoader, {"encoding": "utf8"}), #(GenericLoader.from_filesystem, {"suffixes"=[".py"],"parser"=LanguageParser(language=Language.PYTHON, parser_threshold=500)}),
    ".rst": (UnstructuredRSTLoader, {}),
    # Add more mappings for other file extensions and loaders as needed
}
from code_splitter import python_splitter


def parse_arguments():
    parser = argparse.ArgumentParser(description='ingest: Give as an argument (--storage-folder) the folder in which the embeddings will be stored, '
                                                 'using the power of LLMs.')
    
    # Agrega el argumento --storage-folder
    parser.add_argument('--storage-folder', type=str, help='Folder path for storage')
    
    parser.add_argument('--sumarice', action='store_true', default=False, help='Perform summarization')
    
    
    return parser.parse_args()


def load_single_document(file_path: str) -> Document:
    ext = "." + file_path.rsplit(".", 1)[-1]
    if ext in LOADER_MAPPING:
        loader_class, loader_args = LOADER_MAPPING[ext]
        loader = loader_class(file_path, **loader_args)

    return loader.load()[0]
    raise ValueError(f"Unsupported file extension '{ext}'")


def load_documents(source_dir: str, ignored_files: List[str] = []) -> List[Document]:
    """
    Loads all documents from the source documents directory, ignoring specified files
    """
    import re
    all_files = []
    extensions= []
    for ext in LOADER_MAPPING:
        files=glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
        if len(files)>0:
            extensions.append(ext)
        all_files.extend(files)

        
    filtered_files = [file_path for file_path in all_files if file_path not in ignored_files]

    try:
        pattern = re.compile(r'Tema (\d+)')
        sorted_filtered_files = sorted(filtered_files, key=lambda x: int(pattern.search(x).group(1)))
    except AttributeError as e:
        sorted_filtered_files = filtered_files
        
    print (f"INFORMATION: sorted_filtered_files => {sorted_filtered_files}")
    with Pool(processes=os.cpu_count()) as pool:
        print (f"INFORMATION: Using {os.cpu_count()} as processes")
        results = []
        with tqdm(total=len(sorted_filtered_files), desc='Loading new documents', ncols=80) as pbar:
            for i, doc in enumerate(pool.imap_unordered(load_single_document, sorted_filtered_files)):
                results.append(doc)
                pbar.update()

    print (f"INFORMATION: loaded extensions {extensions}")
    return results, extensions

def get_most_representative_chapters(documents : List[Document], embeddings: HuggingFaceInstructEmbeddings) -> List[int]:
    ## Tenemos que coger todos los capitulos y calcular el Kmedios y calcula el centroide de los clusteres 
    ## de este manera sacaremos los capitulos mas representativos y nos daran una idea general del texto (o textos) 
    '''    
    Load your book into a single text file
    Split your text into large-ish chunks
    Embed your chunks to get vectors
    Cluster the vectors to see which are similar to each other and likely talk about the same parts of the book
    Pick embeddings that represent the cluster the most (method: closest to each cluster centroid)
    Summarize the documents that these embeddings represent
    '''

    vectors = embeddings.embed_documents([doc.page_content for doc in documents])
    # Choose the number of clusters, this can be adjusted based on the book's content.
    # lets say we are only interested in the 10% most relevant information of the book to have a good idea
    num_clusters = int (len(documents) * 0.1)

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(vectors)
    # Find the closest embeddings to the centroids

    # Create an empty list that will hold your closest points
    closest_indices = []

    # Loop through the number of clusters you have
    for i in range(num_clusters):

        # Get the list of distances from that particular cluster center
        distances = np.linalg.norm(vectors - kmeans.cluster_centers_[i], axis=1)

        # Find the list position of the closest one (using argmin to find the smallest distance)
        closest_index = np.argmin(distances)

        # Append that position to your closest indices list
        closest_indices.append(closest_index)
    
    selected_indices = sorted(closest_indices)
    return selected_indices

def filter_and_return_removed(original_list: List, filter_condition: Callable[..., bool]) -> List[Document]:
    """
    Use this function to remove from list of Docs those Docs with specific ending (like .rst or .py)
    and afterward process (RecursiveCharacterTextSplitting) the removed elements. This allow to use
    different splitter for diferent kind of documents. 
    original_list is overwritten
    """
    filtered_list = []
    removed_list = []

    for item in original_list:
        if filter_condition(item):
            filtered_list.append(item)
        else:
            removed_list.append(item)

    original_list = filtered_list
    return removed_list

def process_documents(ignored_files: List[str], summarization: bool, embeddings: HuggingFaceInstructEmbeddings) -> List[Document]:
    """
    Load documents and split in chunks, may raise value error when 0 texts processed
    """
    print(f"INFORMATION: Loading documents from {source_directory}")
    documents, loaded_extensions = load_documents(source_directory, ignored_files)
    if not documents and not summarization:
        print("INFORMATION: No new documents to load and summarization not requested")
        exit(0)
    print(f"INFORMATION: Loaded {len(documents)} new documents from {source_directory}")
    chunk_size = get_perfect_chunk_size(source_dir=source_directory) if summarization else DEFAULT_CHUNK_SIZE
    chunk_overlap = int(chunk_size * 0.3) if summarization else DEFAULT_CHUNK_OVERLAP
    print(f'INFORMATION: Chunk_Size => {chunk_size}  Chunk_Overlap => {chunk_overlap}')
    texts = []
    for extension in loaded_extensions:
        if extension == '.py':
            text_splitter = python_splitter
            print(f"Using splitter for python: {text_splitter}")
        else:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            print(f"Using splitter for text: {text_splitter}")
        texts.extend(text_splitter.split_documents(documents))
        print(f"INFORMATION: Split ext: {extension} into {len(texts)} chunks of text (max. {chunk_size} tokens each)")
    if len(texts) == 0:
        raise ValueError ('No text has been processed')
        
    if summarization and len(texts) > 0:
        most_important_chapter_indexes : List[int] = get_most_representative_chapters(texts, embeddings)
        # if we want a summary of a bunch of texts return most important chunks of text
        most_important_chapters = [texts[chapter_number] for chapter_number in most_important_chapter_indexes]
        if os.path.exists('./'):
            with open('most_important_chapters.pickle', 'wb') as file:
                pickle.dump(most_important_chapters, file)
        else:
            print (f'WARNING: cannot save most important chunks inside {persist_directory}')
        return most_important_chapters
    # Return all chunks
    return texts

def does_vectorstore_exist(persist_directory: str) -> bool:
    """
    Checks if vectorstore exists
    """

    if os.path.exists(os.path.join(persist_directory, 'index')):
        if os.path.exists(os.path.join(persist_directory, 'chroma-collections.parquet')) and os.path.exists(os.path.join(persist_directory, 'chroma-embeddings.parquet')):
            list_index_files = glob.glob(os.path.join(persist_directory, 'index/*.bin'))
            list_index_files += glob.glob(os.path.join(persist_directory, 'index/*.pkl'))
            # At least 3 documents are needed in a working vectorstore
            if len(list_index_files) > 3:
                return True
    return False

def main():
    global persist_directory
    args = parse_arguments()
    if args is not None:
        persist_directory=args.storage_folder
    else: #load persist directory again because after calling parse arguments , if not provided, will be set to none
        persist_directory = os.environ.get('PERSIST_DIRECTORY')
    print (f"INFORMATION: Persist directory: {persist_directory}")    
    # Create embeddings
    embeddings = HuggingFaceInstructEmbeddings(model_name=embeddings_model_name,
                                               model_kwargs={"device": "cuda"})

    if does_vectorstore_exist(persist_directory):
        # Update and store locally vectorstore
        print(f"INFORMATION: Appending to existing vectorstore at {persist_directory}")
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
        collection = db.get()
        try:
            texts = process_documents([metadata['source'] for metadata in collection['metadatas']], args.sumarice, embeddings)
        except ValueError as e:
            print (e)
            exit (1)

        print(f"INFORMATION: Creating embeddings. May take some minutes...")
        db.add_documents(texts)
    else:
        # Create and store locally vectorstore
        print("INFORMATION: Creating new vectorstore")
        try:
            texts = process_documents([], args.sumarice, embeddings)
        except ValueError as e:
            print (e)
            exit (1)

        print(f"INFORMATION: Creating embeddings. May take some minutes...")
        db = Chroma.from_documents(texts, persist_directory=persist_directory, embedding=embeddings, client_settings=CHROMA_SETTINGS)
    db.persist()
    db = None

    print(f"INFORMATION: Ingestion complete! You can now run privateGPT.py to query your documents")


if __name__ == "__main__":
    main()
