from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    Language,
)

python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=1000, chunk_overlap=100
)
