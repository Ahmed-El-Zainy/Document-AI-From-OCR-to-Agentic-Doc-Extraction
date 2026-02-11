# 📄 Document AI: From OCR to Agentic Document Extraction

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![LandingAI](https://img.shields.io/badge/Powered%20by-LandingAI-orange.svg)](https://landing.ai)
[![AWS](https://img.shields.io/badge/AWS-Bedrock-yellow.svg)](https://aws.amazon.com/bedrock/)

*A comprehensive learning journey through modern document intelligence techniques, from traditional OCR to advanced agentic extraction systems*

</div>

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Course Structure](#-course-structure)
- [Prerequisites](#-prerequisites)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Lessons Overview](#-lessons-overview)
- [RAG Pipeline with AWS](#-rag-pipeline-with-aws)
- [Helper Utilities](#-helper-utilities)
- [Project Structure](#-project-structure)
- [Technical Resources](#-technical-resources)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

This repository contains a comprehensive course on **Document AI** and **Intelligent Document Processing (IDP)**, covering the complete evolution from traditional OCR to modern agentic document extraction systems. Learn how to build production-ready document understanding pipelines using cutting-edge technologies.

### What You'll Learn

- 🔍 **Traditional OCR**: Understanding Tesseract and foundational OCR techniques
- 🧠 **Deep Learning OCR**: PaddleOCR and neural network-based text recognition
- 📐 **Layout Analysis**: LayoutLM and LayoutReader for document structure understanding
- 🤖 **Agentic Extraction**: LandingAI's ADE (Agentic Document Extraction)
- ☁️ **Cloud Deployment**: Building RAG pipelines with AWS Bedrock and Lambda
- 💬 **Conversational AI**: Creating document-based chatbots with Strands Agents

---

## ✨ Key Features

| Feature                          | Description                                                    |
| -------------------------------- | -------------------------------------------------------------- |
| **Multi-Modal Processing** | Handle PDFs, images, tables, and complex layouts               |
| **Visual Grounding**       | Maintain bounding box information for precise chunk extraction |
| **Production-Ready**       | AWS Lambda integration for scalable document processing        |
| **RAG Pipeline**           | Complete Retrieval-Augmented Generation system                 |
| **Interactive Learning**   | Jupyter notebooks with hands-on examples                       |
| **Real-World Use Cases**   | Medical documents, invoices, receipts, forms, and more         |

---

## 🏗️ Architecture

### Overall System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Document AI Pipeline                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │      Input Document Processing          │
        │  (PDF, Images, Scanned Documents)       │
        └─────────────────────────────────────────┘
                              │
        ┌─────────────────────┴─────────────────────┐
        │                                           │
        ▼                                           ▼
┌──────────────────┐                    ┌──────────────────┐
│  Traditional OCR │                    │   Deep Learning  │
│    (Tesseract)   │                    │   OCR (Paddle)   │
└──────────────────┘                    └──────────────────┘
        │                                           │
        └─────────────────────┬─────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │       Layout Understanding              │
        │    (LayoutLM, LayoutReader)            │
        └─────────────────────────────────────────┘
                              │
                              ▼
        ┌─────────────────────────────────────────┐
        │      Agentic Document Extraction        │
        │           (LandingAI ADE)               │
        └─────────────────────────────────────────┘
                              │
        ┌─────────────────────┴─────────────────────┐
        │                                           │
        ▼                                           ▼
┌──────────────────┐                    ┌──────────────────┐
│   Structured     │                    │   RAG Pipeline   │
│     Output       │                    │   (AWS Bedrock)  │
│  (Markdown, JSON)│                    │                  │
└──────────────────┘                    └──────────────────┘
                                                  │
                                                  ▼
                                        ┌──────────────────┐
                                        │   Chatbot Agent  │
                                        │ (Strands Agents) │
                                        └──────────────────┘
```

### AWS RAG Pipeline Architecture

```
┌─────────────┐         ┌──────────────┐         ┌─────────────────┐
│   S3 Bucket │────────▶│   Lambda     │────────▶│   LandingAI     │
│ (PDF Upload)│         │  Function    │         │      ADE        │
└─────────────┘         └──────────────┘         └─────────────────┘
                                                           │
                                                           │ Process
                                                           ▼
                                                  ┌─────────────────┐
                                                  │  Extract Chunks │
                                                  │  + Grounding    │
                                                  │  + Metadata     │
                                                  └─────────────────┘
                                                           │
                        ┌──────────────────────────────────┤
                        │                                  │
                        ▼                                  ▼
              ┌─────────────────┐              ┌─────────────────┐
              │ Markdown Output │              │  Chunk JSONs    │
              │  (S3 Storage)   │              │ + Bounding Boxes│
              └─────────────────┘              └─────────────────┘
                                                           │
                                                           ▼
                                                  ┌─────────────────┐
                                                  │     Bedrock     │
                                                  │ Knowledge Base  │
                                                  │   (Vector DB)   │
                                                  └─────────────────┘
                                                           │
                                                           ▼
                                                  ┌─────────────────┐
                                                  │ Strands Agents  │
                                                  │    Chatbot      │
                                                  │ + Visual Ground │
                                                  └─────────────────┘
```

---

## 📚 Course Structure

The course is organized into progressive lessons, each building upon previous concepts:

| Lesson          | Topic               | Technologies                 | Difficulty          |
| --------------- | ------------------- | ---------------------------- | ------------------- |
| **L1**    | Introduction to OCR | Tesseract                    | ⭐ Beginner         |
| **L2**    | Document Processing | Tesseract, PaddleOCR         | ⭐⭐ Beginner       |
| **L3**    | Layout Analysis     | LayoutLM                     | ⭐⭐ Intermediate   |
| **L4**    | Advanced OCR        | PaddleOCR                    | ⭐⭐ Intermediate   |
| **L6**    | Reading Order       | LayoutReader                 | ⭐⭐⭐ Intermediate |
| **L8**    | Agentic Extraction  | LandingAI ADE                | ⭐⭐⭐ Advanced     |
| **L9**    | Batch Processing    | LandingAI ADE                | ⭐⭐⭐ Advanced     |
| **L11**   | RAG with ChromaDB   | ChromaDB, LangChain          | ⭐⭐⭐⭐ Advanced   |
| **Lab 6** | AWS RAG Pipeline    | AWS Bedrock, Lambda, Strands | ⭐⭐⭐⭐⭐ Expert   |

---

## 🔧 Prerequisites

### System Requirements

- **Python**: Version 3.10 (recommended)
- **OS**: Linux, macOS, or Windows (Linux x86_64 recommended for AWS Lambda)
- **Memory**: 8GB RAM minimum (16GB recommended)
- **Storage**: 5GB free space

### Required Accounts

1. **LandingAI Account** (Free tier available)

   - Sign up at [LandingAI](https://landing.ai)
   - Get your Vision Agent API key
2. **AWS Account** (for Lab 6 only)

   - Required services: S3, Lambda, Bedrock, IAM
   - Estimated cost: ~$5-10/month for testing
3. **OpenAI Account** (optional, for advanced features)

---

## 📦 Installation

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd document_ai_from_OCR_to_agentic_doc_extraction
```

### Step 2: Create Virtual Environment

```bash
# Using venv
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n docai python=3.10
conda activate docai
```

### Step 3: Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# For AWS Lab 6 (optional)
pip install boto3 bedrock-agentcore strands-agents
```

### Step 4: Install System Dependencies

#### For Tesseract OCR (L1, L2):

```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract

# Windows
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
```

#### For PaddleOCR (L2, L4):

```bash
pip install paddlepaddle==3.0.0 paddleocr
```

### Step 5: Configure Environment Variables

Create a `.env` file in the project root:

```bash
# LandingAI Configuration
VISION_AGENT_API_KEY=your_landingai_api_key_here

# AWS Configuration (for Lab 6)
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=us-west-2
S3_BUCKET=your-bucket-name
BEDROCK_MODEL_ID=us.anthropic.claude-sonnet-4-5-20250929-v1:0
BEDROCK_KB_ID=your_knowledge_base_id

# OpenAI Configuration (optional)
OPENAI_API_KEY=your_openai_api_key
```

---

## 🚀 Quick Start

### Example 1: Basic OCR with Tesseract (Lesson 2)

```python
import pytesseract
from PIL import Image

# Load and process image
image = Image.open("L2/invoice.png")
text = pytesseract.image_to_string(image)
print(text)
```

### Example 2: Advanced OCR with PaddleOCR (Lesson 4)

```python
from paddleocr import PaddleOCR

# Initialize OCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Process document
result = ocr.ocr('L4/bank_statement.png', cls=True)

# Extract text
for line in result:
    for word_info in line:
        print(word_info[1][0])  # Extracted text
```

### Example 3: Agentic Document Extraction (Lesson 8)

```python
from landingai.ade import ADEClient

# Initialize ADE client
client = ADEClient(api_key="your_api_key")

# Parse document with visual grounding
response = client.parse(
    document_path="document.pdf",
    extract_tables=True,
    extract_figures=True
)

# Access structured output
markdown_content = response.markdown
groundings = response.grounding  # Bounding box information

print(markdown_content)
```

### Example 4: RAG Pipeline Query (Lesson 11)

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Load vector database
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=OpenAIEmbeddings()
)

# Query documents
results = vectorstore.similarity_search(
    "What are the company's revenue figures?",
    k=5
)

for doc in results:
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}\n")
```

---

## 📖 Lessons Overview

### Lesson 2: Introduction to Document Processing

**Focus**: Traditional OCR with Tesseract and PaddleOCR

📁 **Directory**: `L2/`

- `L2.ipynb` - Main tutorial notebook
- `invoice.png`, `receipt.jpg`, `table.png` - Sample documents

**Key Concepts**:

- Image preprocessing techniques
- Text extraction from various document types
- Handling tables and forms
- Comparing Tesseract vs. PaddleOCR performance

**Use Cases**:

- ✅ Simple invoices and receipts
- ✅ Clean, scanned documents
- ⚠️ Limited table structure recognition
- ❌ Complex layouts not well supported

---

### Lesson 4: Advanced OCR with PaddleOCR

**Focus**: Deep learning-based OCR with better accuracy

📁 **Directory**: `L4/`

- `L4.ipynb` - Advanced OCR techniques
- `bank_statement.png`, `handwritten.jpg` - Complex documents

**Key Concepts**:

- Neural network-based text detection
- Multi-language support
- Angle classification for rotated text
- Confidence scoring

**Improvements over L2**:

- ✅ Better handling of curved or rotated text
- ✅ Improved accuracy on low-quality scans
- ✅ Multi-language text recognition
- ✅ Handwriting recognition support

---

### Lesson 6: Layout Analysis with LayoutReader

**Focus**: Understanding document structure and reading order

📁 **Directory**: `L6/`

- `L6.ipynb` - Layout understanding tutorial
- `layoutreader/` - LayoutReader implementation
- `architecture.png`, `report_layout.png` - Visualization examples

**Key Concepts**:

- Document layout analysis
- Reading order determination
- Relationship between text blocks
- Visual structure recognition

**Architecture**:

```
Document Image
      │
      ▼
┌──────────────┐
│ Layout Model │  (LayoutLM/LayoutLMv2)
└──────────────┘
      │
      ▼
┌──────────────┐
│   Bounding   │  (Text blocks, tables, figures)
│    Boxes     │
└──────────────┘
      │
      ▼
┌──────────────┐
│ Reading Order│  (Sequence prediction)
│ Determination│
└──────────────┘
      │
      ▼
Structured Output
```

---

### Lesson 8: Agentic Document Extraction

**Focus**: Modern AI-powered document understanding with LandingAI ADE

📁 **Directory**: `L8/`

- `L8.ipynb` - ADE comprehensive tutorial
- `helper.py` - Visualization utilities
- `utility_example/` - Advanced examples
- `difficult_examples/` - Edge cases

**Key Concepts**:

- Agentic approach to document extraction
- Automatic chunk detection (text, tables, figures)
- Visual grounding with bounding boxes
- Markdown output with preserved structure
- Confidence scoring for extractions

**Chunk Types**:

- 📝 `chunkText` - Regular text paragraphs
- 📊 `chunkTable` - Structured tables
- 🖼️ `chunkFigure` - Images and diagrams
- 🏷️ `chunkLogo` - Company logos
- 📇 `chunkCard` - Business cards
- ✍️ `chunkAttestation` - Signatures
- 📱 `chunkScanCode` - QR/Barcodes
- 📋 `chunkForm` - Form fields

**Visualization Example**:

```python
from helper import draw_bounding_boxes

# Parse document
response = ade_client.parse("document.pdf")

# Draw bounding boxes on chunks
draw_bounding_boxes(response, "document.pdf")
```

**Output**: Color-coded bounding boxes showing:

- 🟢 Green: Text chunks
- 🔵 Blue: Tables
- 🟣 Purple: Marginalia
- 🟠 Orange: Cards

---

### Lesson 9: Batch Processing with ADE

**Focus**: Processing multiple documents efficiently

📁 **Directory**: `L9/`

- `L9.ipynb` - Batch processing workflow
- `input_folder/` - Sample documents for batch processing
- `results/` - Processed outputs
- `results_extracted/` - Extracted structured data

**Key Concepts**:

- Batch document processing
- Parallel processing strategies
- Error handling and logging
- Output organization
- Performance optimization

**Workflow**:

```python
import os
from pathlib import Path
from landingai.ade import ADEClient

client = ADEClient(api_key=os.getenv("VISION_AGENT_API_KEY"))
input_dir = Path("input_folder")
output_dir = Path("results")

for doc_path in input_dir.glob("*.pdf"):
    try:
        response = client.parse(doc_path)
  
        # Save markdown
        (output_dir / f"{doc_path.stem}.md").write_text(response.markdown)
  
        # Save grounding data
        # ... (save JSON with bounding boxes)
  
        print(f"✅ Processed: {doc_path.name}")
    except Exception as e:
        print(f"❌ Failed: {doc_path.name} - {e}")
```

---

### Lesson 11: RAG with ChromaDB

**Focus**: Building a Retrieval-Augmented Generation system

📁 **Directory**: `L11/`

- `L11.ipynb` - RAG implementation tutorial
- `apple_10k.pdf` - Sample financial document
- `chroma_db/` - Vector database storage
- `ade_outputs/` - Processed document chunks

**Key Concepts**:

- Document chunking strategies
- Vector embeddings
- Semantic search
- Context retrieval
- LLM integration for Q&A

**RAG Pipeline Flow**:

```
Document (PDF)
      │
      ▼
  ADE Parse  ──────> Markdown + Grounding
      │
      ▼
  Chunking   ──────> Semantic segments
      │
      ▼
  Embeddings ──────> Vector representations
      │
      ▼
  ChromaDB   ──────> Vector storage
      │
      ▼
  User Query
      │
      ▼
 Similarity  ──────> Retrieve relevant chunks
   Search
      │
      ▼
  LLM + Context ───> Generate answer
```

**Example Query Flow**:

```python
# 1. Load and parse document
response = ade_client.parse("apple_10k.pdf")

# 2. Create chunks
chunks = create_semantic_chunks(response.markdown)

# 3. Store in vector DB
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=OpenAIEmbeddings(),
    persist_directory="./chroma_db"
)

# 4. Query
query = "What was Apple's revenue in 2023?"
docs = vectorstore.similarity_search(query, k=5)

# 5. Generate answer with LLM
context = "\n\n".join([doc.page_content for doc in docs])
answer = llm.invoke(f"Context: {context}\n\nQuestion: {query}")
```

---

## ☁️ RAG Pipeline with AWS

### Overview

Lab 6 demonstrates a production-ready document intelligence system using AWS services.

📁 **Directory**: `rag_pipeline_aws/`

### Architecture Components

```
┌─────────────────────────────────────────────────────────────────┐
│                    AWS Cloud Infrastructure                      │
└─────────────────────────────────────────────────────────────────┘

┌──────────────┐
│  User Upload │
│  PDF to S3   │
└──────┬───────┘
       │
       ▼
┌─────────────────────────────────────────────────────┐
│              S3 Bucket Structure                    │
├─────────────────────────────────────────────────────┤
│  input/                                             │
│    └── medical/                                     │
│         └── research_papers.pdf                     │
│                                                     │
│  output/                                            │
│    ├── medical/                  (Markdown)         │
│    ├── medical_grounding/        (Bounding boxes)   │
│    ├── medical_chunks/           (Chunk JSONs)      │
│    └── medical_chunk_images/     (Cropped images)   │
└─────────────────────────────────────────────────────┘
       │
       │ S3 Event Trigger
       ▼
┌──────────────────────────────────────────┐
│           Lambda Function                │
│      (ade_s3_handler.py)                 │
├──────────────────────────────────────────┤
│  • Triggered on S3 upload                │
│  • Calls LandingAI ADE API               │
│  • Processes document                    │
│  • Creates chunk JSONs                   │
│  • Saves to S3 output/                   │
└──────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────┐
│       AWS Bedrock Knowledge Base         │
├──────────────────────────────────────────┤
│  • Indexes chunk JSONs                   │
│  • Maintains metadata                    │
│  • Vector embeddings                     │
│  • Semantic search                       │
└──────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────┐
│         Strands Agent Framework          │
├──────────────────────────────────────────┤
│  • Orchestrates conversation             │
│  • Queries Knowledge Base                │
│  • Visual grounding tool                 │
│  • Bedrock Memory Service                │
└──────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────┐
│          User Interaction                │
│  • Ask questions about documents         │
│  • Get answers with source citations     │
│  • View highlighted document regions     │
└──────────────────────────────────────────┘
```

### Key Files

- `Lab-6.ipynb` - Main tutorial notebook
- `ade_s3_handler.py` - Lambda function for document processing
- `lambda_helpers.py` - Deployment utilities
- `visual_grounding_helper.py` - Chunk image extraction
- `medical/` - Sample medical research papers

### Quick Setup

```bash
# 1. Configure AWS credentials
aws configure

# 2. Create S3 bucket
aws s3 mb s3://your-doc-bucket
aws s3api put-object --bucket your-doc-bucket --key input/
aws s3api put-object --bucket your-doc-bucket --key output/

# 3. Deploy Lambda (see Lab-6.ipynb for details)
# 4. Create Bedrock Knowledge Base
# 5. Upload documents and start chatting!
```

For detailed setup instructions, see [rag_pipeline_aws/README.md](rag_pipeline_aws/README.md)

---

## 🛠️ Helper Utilities

The `helper.py` file provides essential utilities for document visualization and processing.

### Key Functions

#### 1. Document Display

```python
from helper import print_document

# Display PDF or image in notebook
print_document("document.pdf")
print_document("image.png")
```

#### 2. Bounding Box Visualization

```python
from helper import draw_bounding_boxes

# Draw color-coded bounding boxes
parse_response = ade_client.parse("document.pdf")
annotated_image = draw_bounding_boxes(parse_response, "document.pdf")
```

**Color Scheme**:

- 🟢 **Green** (40, 167, 69): Text chunks (`chunkText`)
- 🔵 **Blue** (0, 123, 255): Tables (`chunkTable`)
- 🟣 **Purple** (111, 66, 193): Marginalia (`chunkMarginalia`)
- 🟡 **Magenta** (255, 0, 255): Figures (`chunkFigure`)
- 🟢 **Light Green** (144, 238, 144): Logos (`chunkLogo`)
- 🟠 **Orange** (255, 165, 0): Cards (`chunkCard`)
- 🔵 **Cyan** (0, 255, 255): Attestations (`chunkAttestation`)
- 🟡 **Yellow** (255, 193, 7): Scan codes (`chunkScanCode`)
- 🔴 **Red** (220, 20, 60): Forms (`chunkForm`)

---

## 📂 Project Structure

```
document_ai_from_OCR_to_agentic_doc_extraction/
│
├── README.md                      # This comprehensive guide
├── requirements.txt               # Python dependencies
├── helper.py                      # Global utility functions
├── .env                          # Environment variables (not in git)
├── .gitignore                    # Git ignore rules
│
├── L2/                           # Lesson 2: Basic OCR
│   ├── L2.ipynb                  # Jupyter notebook
│   ├── l2_doc_processing.py      # Python utilities
│   ├── invoice.png               # Sample invoice
│   ├── receipt.jpg               # Sample receipt
│   ├── table.png                 # Sample table
│   └── requirements.txt          # Lesson-specific deps
│
├── L4/                           # Lesson 4: PaddleOCR
│   ├── L4.ipynb
│   ├── l4_doc_parsing_paddleocr.py
│   ├── bank_statement.png
│   ├── handwritten.jpg
│   └── article.jpg
│
├── L6/                           # Lesson 6: Layout Analysis
│   ├── L6.ipynb
│   ├── architecture.png
│   ├── report_layout.png
│   └── layoutreader/             # LayoutReader implementation
│       ├── README.md
│       ├── main.py
│       └── tools.py
│
├── L8/                           # Lesson 8: Agentic Extraction
│   ├── L8.ipynb
│   ├── helper.py
│   ├── difficult_examples/       # Complex document samples
│   └── utility_example/
│
├── L9/                           # Lesson 9: Batch Processing
│   ├── L9.ipynb
│   ├── helper.py
│   ├── input_folder/             # Documents to process
│   ├── results/                  # Markdown outputs
│   └── results_extracted/        # Structured extractions
│
├── L11/                          # Lesson 11: RAG Pipeline
│   ├── L11.ipynb
│   ├── helper.py
│   ├── apple_10k.pdf             # Sample financial document
│   ├── ade_outputs/
│   └── chroma_db/                # Vector database
│
└── rag_pipeline_aws/             # Lab 6: AWS RAG System
    ├── Lab-6.ipynb
    ├── README.md                 # Detailed lab guide
    ├── ade_s3_handler.py         # Lambda function
    ├── lambda_helpers.py         # Deployment tools
    ├── visual_grounding_helper.py # Chunk extraction
    └── medical/                   # Sample medical PDFs
```

---

## 📚 Technical Resources

### Core Technologies

#### OCR Engines

- **Tesseract**

  - [Official Documentation](https://tesseract-ocr.github.io/)
  - [Technical Report](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/33418.pdf)
  - Best for: Clean, printed text
- **PaddleOCR**

  - [GitHub Repository](https://github.com/PaddlePaddle/PaddleOCR)
  - [Technical Report](https://arxiv.org/abs/2507.05595)
  - Best for: Complex layouts, multilingual, handwriting

#### Layout Understanding

- **LayoutLM**

  - [Technical Report](https://arxiv.org/abs/1912.13318)
  - [Hugging Face Models](https://huggingface.co/models?search=layoutlm)
  - Capabilities: Document understanding with visual + text + layout
- **LayoutReader**

  - [Technical Report](https://arxiv.org/abs/2108.11591)
  - Capabilities: Reading order prediction

#### LandingAI Platform

- **Vision Agent**
  - [Platform](https://va.landing.ai/)
  - [Documentation](https://docs.landing.ai/)
  - [Blog: Evolution of OCR](https://landing.ai/blog/ocr-to-agentic-document-extraction-a-look-into-the-evolution-of-document-intelligence/)
  - [Blog: DocVQA Performance](https://landing.ai/blog/superhuman-on-docvqa-without-images-in-qa-agentic-document-extraction/)

#### AWS Services

- **S3** - [Documentation](https://docs.aws.amazon.com/s3/)
- **Lambda** - [Documentation](https://docs.aws.amazon.com/lambda/)
- **IAM** - [Documentation](https://docs.aws.amazon.com/iam/)
- **Bedrock** - [Documentation](https://docs.aws.amazon.com/bedrock/)

#### Python Libraries

- **boto3** - [AWS SDK](https://docs.aws.amazon.com/pythonsdk/)
- **bedrock-agentcore** - [Documentation](https://docs.aws.amazon.com/bedrock-agentcore/)
- **strands-agents** - [Guide](https://docs.aws.amazon.com/prescriptive-guidance/latest/agentic-ai-frameworks/strands-agents.html)
  - [Blog Post](https://aws.amazon.com/blogs/opensource/introducing-strands-agents-an-open-source-ai-agents-sdk/)

---

## 🔍 Troubleshooting

### Common Issues

#### 1. Tesseract Not Found

**Error**: `TesseractNotFoundError`

**Solution**:

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install tesseract-ocr

# Verify installation
tesseract --version

# If needed, specify path
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
```

#### 2. PaddlePaddle Installation Issues

**Error**: `No module named 'paddle'`

**Solution**:

```bash
# Uninstall any existing version
pip uninstall paddlepaddle paddlepaddle-gpu

# Install correct version
pip install paddlepaddle==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/

# For GPU (CUDA 11.2)
pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu112/
```

#### 3. LandingAI API Key Issues

**Error**: `Authentication failed`

**Solution**:

```bash
# Verify .env file exists
cat .env | grep VISION_AGENT_API_KEY

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Or set directly (not recommended for production)
import os
os.environ['VISION_AGENT_API_KEY'] = 'your_key_here'
```

#### 4. AWS Lambda Timeout

**Error**: `Task timed out after 3.00 seconds`

**Solution**:

```python
# Increase Lambda timeout
lambda_client.update_function_configuration(
    FunctionName='doc-processor',
    Timeout=900,  # 15 minutes
    MemorySize=1024  # 1GB RAM
)
```

---

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### Ways to Contribute

1. **🐛 Report Bugs** - Use GitHub Issues
2. **✨ Suggest Features** - Propose new lessons or examples
3. **📖 Improve Documentation** - Fix typos, add clarifications
4. **💻 Submit Code** - Fork, create feature branch, submit PR

### Development Setup

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/document_ai_from_OCR_to_agentic_doc_extraction.git
cd document_ai_from_OCR_to_agentic_doc_extraction

# Create branch
git checkout -b feature/your-feature-name

# Make changes and commit
git commit -m "Add: Brief description of changes"

# Push and create PR
git push origin feature/your-feature-name
```

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- **DeepLearning.AI** for the course structure
- **LandingAI** for the ADE platform and Vision Agent
- **AWS** for cloud infrastructure support
- **PaddlePaddle** team for PaddleOCR
- **Microsoft** for LayoutLM research
- **Google** for Tesseract OCR

---

## 📞 Support

- 📧 **LandingAI Support**: support@landingai.com
- 📖 **Documentation**: [docs.landing.ai](https://docs.landing.ai)
- 🐛 **Issues**: GitHub Issues

---

<div align="center">

**[⬆ Back to Top](#-document-ai-from-ocr-to-agentic-document-extraction)**

---

*Last Updated: February 2026*

</div>
