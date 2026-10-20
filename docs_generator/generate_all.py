"""
Master Documentation Assembler:
1. Compiles Chapters 1 to 16 into PROJECT_DOCUMENTATION_AND_LEARNING_GUIDE.md
2. Builds PROJECT_DOCUMENTATION_AND_LEARNING_GUIDE.docx with professional formatting
"""

import sys
from pathlib import Path

# Add current directory to path
BASE_DIR = Path(r"D:\Brain_Tumor")
sys.path.insert(0, str(BASE_DIR))

from docs_generator.chapter_1_to_3 import CHAPTER_1, CHAPTER_2, CHAPTER_3
from docs_generator.chapter_4_to_5 import CHAPTER_4, CHAPTER_5
from docs_generator.chapter_6_to_7 import CHAPTER_6, CHAPTER_7
from docs_generator.chapter_8_to_9 import CHAPTER_8, CHAPTER_9
from docs_generator.chapter_10_to_12 import CHAPTER_10, CHAPTER_11, CHAPTER_12
from docs_generator.chapter_13_to_14 import CHAPTER_13, CHAPTER_14
from docs_generator.chapter_15_to_16 import CHAPTER_15, CHAPTER_16
from docs_generator.docx_builder import create_styled_document, add_title_block, parse_markdown_to_docx

def main():
    print("=" * 60)
    print("NEUROSCAN AI: DOCUMENTATION & LEARNING GUIDE GENERATOR")
    print("=" * 60)
    
    # 1. Combine All Chapters
    chapters = [
        CHAPTER_1, CHAPTER_2, CHAPTER_3,
        CHAPTER_4, CHAPTER_5,
        CHAPTER_6, CHAPTER_7,
        CHAPTER_8, CHAPTER_9,
        CHAPTER_10, CHAPTER_11, CHAPTER_12,
        CHAPTER_13, CHAPTER_14,
        CHAPTER_15, CHAPTER_16
    ]
    
    full_markdown_text = "\n\n---\n\n".join(chapters)
    # Sanitize accidental escape control characters from LaTeX math backslashes
    full_markdown_text = full_markdown_text.replace('\x07', r'\a').replace('\x0c', r'\f').replace('\x08', r'\b')
    
    # 2. Write Markdown File
    md_file_path = BASE_DIR / "PROJECT_DOCUMENTATION_AND_LEARNING_GUIDE.md"
    print(f"Writing Markdown documentation to:\n{md_file_path}")
    with open(md_file_path, "w", encoding="utf-8") as f:
        f.write(full_markdown_text)
    print(f"Markdown file written successfully! ({len(full_markdown_text)} characters, {len(full_markdown_text.split())} words)\n")
    
    # 3. Build DOCX Document
    docx_file_path = BASE_DIR / "PROJECT_DOCUMENTATION_AND_LEARNING_GUIDE.docx"
    print(f"Building Word document (.docx) at:\n{docx_file_path}...")
    doc = create_styled_document()
    
    # Title Block
    add_title_block(
        doc,
        title="PROJECT DOCUMENTATION & MASTER LEARNING GUIDE",
        subtitle="NeuroScan AI — Brain Tumor Detection & Explainable AI (Grad-CAM)",
        author="Aditya Sarapure",
        project_name="NeuroScan AI Medical Diagnostic System"
    )
    
    # Parse Markdown into Styled Word Elements
    parse_markdown_to_docx(doc, full_markdown_text)
    
    # Save DOCX
    doc.save(str(docx_file_path))
    print(f"Word document saved successfully! File size: {docx_file_path.stat().st_size / 1024:.1f} KB")
    print("=" * 60)
    print("DOCUMENTATION GENERATION COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    main()
