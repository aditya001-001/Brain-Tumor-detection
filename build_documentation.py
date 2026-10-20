"""
Script to generate:
1. PROJECT_DOCUMENTATION_AND_LEARNING_GUIDE.md
2. PROJECT_DOCUMENTATION_AND_LEARNING_GUIDE.docx

Complete End-to-End Documentation and Learning Guide for NeuroScan AI.
"""

import os
from pathlib import Path
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

BASE_DIR = Path(r"D:\Brain_Tumor")
MD_PATH = BASE_DIR / "PROJECT_DOCUMENTATION_AND_LEARNING_GUIDE.md"
DOCX_PATH = BASE_DIR / "PROJECT_DOCUMENTATION_AND_LEARNING_GUIDE.docx"

print("Starting documentation generation...")
