#!/usr/bin/env python3
import os
import glob
from typing import List
from dotenv import load_dotenv
from multiprocessing import Pool
from tqdm import tqdm
import pickle

from langchain.llms import GPT4All, LlamaCpp
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains.summarize import load_summarize_chain
from langchain import PromptTemplate
from langchain.memory import ConversationSummaryBufferMemory
from langchain.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PDFMinerLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)
import pdb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings
from langchain.docstore.document import Document
from constants import CHROMA_SETTINGS

load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX_SUMMARY')

from constants import CHROMA_SETTINGS


map_prompt_es = """
Vas a recibir un capitulo de un libro. Este capitulo estara entre tres backticks (```)
Tu objetivo es producir un RESUMEN de este capitulo para que el lector pueda tener un entendimiento completo de lo que ha pasado
Su respuesta debe constar de al menos tres párrafos y abarcar íntegramente lo dicho en el pasaje sin frases sin terminar

```{text}```
RESUMEN:
"""

combine_prompt = """
Se le darán una serie de resúmenes de un libro. Los resúmenes irán entre tres backticks (```).
Su objetivo es resumir de forma verborreica lo sucedido en la historia.
El lector debe ser capaz de comprender lo que ocurrió en el libro.

```{text}```
RESUMEN DETALLADO:
"""



map_prompt_en = """
You will be given a single passage of a book. This section will be enclosed in triple backticks (```)
Your goal is to give a summary of this section so that a reader will have a full understanding of what happened.
Your response should be at least three paragraphs and fully encompass what was said in the passage.

```{text}```
FULL SUMMARY:
"""

map_prompt = {'en' : map_prompt_en, 'es' : map_prompt_es }

def display_summary(docs : List[Document], language: str, llm) -> None:
    """Display a summary of texts given as input

    Args:
        texts (List[Document]): _description_
        llm (Any[LlamaCpp, GPT4All]): _description_
    """

    import textwrap
    
    from typing import Dict, Any
    
    def my_save_context(self, inputs: Document, outputs: Document) -> None:
        """Save context from this conversation to buffer."""
        super().save_context(inputs.page_content, outputs.page_content)
        self.prune()
    ConversationSummaryBufferMemory.save_context = my_save_context
    memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=700)
    
    combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])
    
    map_prompt_template = PromptTemplate(template=map_prompt[language], input_variables=["text"])
    map_chain = load_summarize_chain(llm=llm,
                             chain_type="stuff",
                             #memory = memory,
                             prompt=map_prompt_template)
    
    
    #summary_chain = load_summarize_chain(llm=llm,
    #                                 chain_type='map_reduce',
    #                                 map_prompt=map_prompt_template,
    #                                 combine_prompt=combine_prompt_template,
    #                                   verbose=True
    #                                )

    #output = summary_chain.run(docs)
    #print (output)
    #return
    # Make an empty list to hold your summaries
    summary_list = []

    # Loop through a range of the lenght of your selected docs
    chapter: int  = 0
    for document in docs:
        
        # Go get a summary of the chunk
        chunk_summary = map_chain.run([document])
        
        # Append that summary to your list
        summary_list.append(chunk_summary)
        
        print (f"Summary #{chapter} (chunk #) - Preview: {chunk_summary[:250]} \n")
        chapter+=1

    summaries = "\n".join(summary_list)

    # Convert it back to a document
    summaries = Document(page_content=summaries)

    print (f"Your total summary has {llm.get_num_tokens(summaries.page_content)} tokens")
    with open('private_summaries.pkl', 'wb') as file:
        pickle.dump(summaries, file)
    
    combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])
    reduce_chain = load_summarize_chain(llm=llm,
                                 chain_type="stuff",
                                 prompt=combine_prompt_template,
                                 memory=memory,
                                 verbose=True # Set this to true if you want to see the inner workings
                                   )
    
    for summary in summary_list:    
        output = reduce_chain.run([Document(page_content=summary)])
        
    print ("RESULTADO_FINAL:")
    print (output)



callbacks = [StreamingStdOutCallbackHandler()]
# Prepare the LLM
match model_type:
    case "LlamaCpp":
        llm = LlamaCpp(model_path=model_path, n_batch=4096, temperature=0.0, n_ctx=model_n_ctx, callbacks=callbacks, verbose=False)
    case "GPT4All":
        llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', callbacks=callbacks, verbose=False)
    case _default:
        print(f"Model {model_type} not supported!")
        exit(1);

file = open('most_important_chapters.pickle', 'rb')
loaded_data = pickle.load(file)



text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
texts = text_splitter.split_documents(loaded_data)


display_summary(texts,'en', llm)
    
#
#for idex, doc in enumerate(loaded_data):
#    with open(f'capitulo_{idex}.txt', 'w') as file:
#        file.write(doc.page_content)
#
