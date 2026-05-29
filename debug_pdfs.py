# Save as debug_pdfs.py and run it separately: python debug_pdfs.py

import os
from langchain_community.document_loaders import PyPDFLoader

PDF_FOLDER = r"C:\Books&Notes\Class IX\Science"   # your actual path

pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]

for filename in pdf_files:
    path = os.path.join(PDF_FOLDER, filename)
    loader = PyPDFLoader(path)
    pages = loader.load()

    print(f"\n{'='*50}")
    print(f"FILE: {filename}")
    print(f"Pages loaded: {len(pages)}")

    # Check first 3 pages for actual text content
    for i, page in enumerate(pages[:3]):
        text = page.page_content.strip()
        print(f"\n  Page {i+1}: {len(text)} characters")
        if len(text) < 50:
            print(f"  ⚠️  WARNING: Very little text — may be scanned/image PDF")
        else:
            print(f"  ✓ Sample: '{text[:150]}...'")