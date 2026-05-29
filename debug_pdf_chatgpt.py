import os
from langchain_community.document_loaders import PyPDFLoader

PDF_FOLDER = r"C:\Books\Class IX\Science"

print("Script started")

print("Folder exists:", os.path.exists(PDF_FOLDER))

pdf_files = [f for f in os.listdir(PDF_FOLDER)
             if f.lower().endswith(".pdf")]

print("PDF files found:", pdf_files)

for filename in pdf_files:

    print(f"\nTrying to load: {filename}")

    try:
        path = os.path.join(PDF_FOLDER, filename)

        loader = PyPDFLoader(path)

        pages = loader.load()

        print(f"SUCCESS: {filename}")
        print(f"Pages: {len(pages)}")

    except Exception as e:
        print(f"ERROR loading {filename}")
        print(e)