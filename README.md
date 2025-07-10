# RAG Book - Vietnamese High School Textbook RAG System

A comprehensive Retrieval-Augmented Generation (RAG) system designed specifically for Vietnamese high school textbooks. This project provides intelligent question-answering capabilities for educational content across multiple subjects and grade levels.

## 🎯 Project Overview

This RAG system is built to help students and educators access and query Vietnamese high school textbook content efficiently. It supports multiple textbook series including:

- **Kết nối tri thức** (Knowledge Connection)
- **Chân trời sáng tạo** (Creative Horizon) 
- **Cánh diều** (Kite)

### Supported Subjects
- Mathematics (Toán học)
- Literature (Ngữ văn)
- Physics (Vật lý)
- Chemistry (Hóa học)
- Biology (Sinh học)
- History (Lịch sử)
- Geography (Địa lý)
- English (Tiếng Anh)

### Grade Levels
- Grade 10 (Lớp 10)
- Grade 11 (Lớp 11)
- Grade 12 (Lớp 12)

## 🚀 Features

### Core RAG Capabilities
- **Baseline RAG**: Standard retrieval and generation using vector search
- **Agentic RAG**: Advanced sub-question decomposition for complex queries
- **Multi-language Support**: Optimized for Vietnamese content with English support
- **Persistent Indexing**: Efficient storage and retrieval of document embeddings

### Document Processing
- **PDF to Markdown Conversion**: Automated conversion using LlamaParse
- **Layout-Aware Parsing**: Preserves document structure and formatting
- **Mathematical Equation Support**: LaTeX formatting for mathematical content
- **Image Description**: AI-powered descriptions of visual content

### Advanced Features
- **Sub-question Decomposition**: Breaks complex questions into simpler sub-questions
- **Contextual Search**: Intelligent retrieval based on question context
- **Multi-document Queries**: Searches across multiple textbooks simultaneously

## 📋 Prerequisites

- Python 3.13 or higher
- OpenAI API key
- UV package manager (recommended)

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd rag-book
   ```

2. **Set up environment**
   ```bash
   # Install UV if not already installed
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Create virtual environment and install dependencies
   uv sync
   ```

3. **Configure API keys**
   ```bash
   # Copy the setup script and modify with your API key
   cp setup.sh setup.sh.backup
   # Edit setup.sh with your OpenAI API key
   source setup.sh
   ```

   Or create a `.env` file:
   ```bash
   echo "OPENAI_API_KEY=your_api_key_here" > .env
   ```

## 📚 Data Processing

### Converting PDFs to Markdown

The project includes a script to convert PDF textbooks to markdown format:

```bash
# Convert a single PDF file
python src/scripts/convert_to_markdown.py input_folder output_folder --mode baseline

# Available modes:
# - baseline: Standard parsing
# - lvm: Layout-aware parsing with vision model
```

### Supported Parsing Modes

1. **Baseline Mode**: Standard LlamaParse for general documents
2. **LVM Mode**: Layout-aware parsing using vision models for better structure preservation
3. **English Mode**: Specialized parsing for English textbooks

## 🔧 Usage

### Basic RAG Query

```python
from src.rag.base_rag import LocalBaselineRAG

# Initialize RAG system
rag = LocalBaselineRAG()

# Ask a question
question = "Giải thích về định luật Newton thứ nhất"
answer = rag.get_answer(question)
print(answer)
```

### Agentic RAG for Complex Queries

```python
from src.rag.base_rag import LocalSubQuestionRAG

# Initialize agentic RAG system
rag = LocalSubQuestionRAG()

# Ask a complex question that requires multiple steps
question = "So sánh các phương pháp giải phương trình bậc hai trong sách giáo khoa"
answer = rag.get_answer(question)
print(answer)
```

### Jupyter Notebooks

The project includes interactive notebooks for experimentation:

- `src/notebooks/baseline_rag.ipynb`: Basic RAG functionality
- `src/notebooks/agentic_rag.ipynb`: Advanced agentic RAG features

## 🏗️ Project Structure

```
rag-book/
├── src/
│   ├── rag/
│   │   └── base_rag.py          # Core RAG implementation
│   ├── scripts/
│   │   └── convert_to_markdown.py  # PDF to markdown converter
│   ├── utils/
│   │   ├── parse_config.py      # LlamaParse configurations
│   │   ├── eval.py             # Evaluation utilities
│   │   └── common.py           # Common utilities
│   ├── notebooks/              # Jupyter notebooks
│   ├── data/                   # Processed markdown files
│   ├── raw/                    # Raw PDF files
│   └── .index_storage/         # Vector index storage
├── pyproject.toml              # Project configuration
├── uv.lock                     # Dependency lock file
└── setup.sh                    # Environment setup
```

## 🔍 RAG Architecture

### Base RAG Classes

1. **BaseRAG**: Abstract base class defining RAG interface
2. **LocalRAG**: Local implementation with OpenAI integration
3. **LocalBaselineRAG**: Standard query engine implementation
4. **LocalSubQuestionRAG**: Advanced sub-question decomposition

### Key Components

- **Vector Store Index**: Built using LlamaIndex for efficient document retrieval
- **Query Engine**: Handles question processing and answer generation
- **Sub-question Engine**: Decomposes complex questions into simpler queries
- **Document Tools**: Specialized tools for document search and retrieval

## 🧪 Evaluation

The project includes evaluation utilities in `src/utils/eval.py` for assessing RAG performance on Vietnamese educational content.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with [LlamaIndex](https://github.com/run-llama/llama_index) for RAG capabilities
- Uses [LlamaParse](https://github.com/run-llama/llama_parse) for document processing
- Powered by OpenAI's GPT models for text generation
- Designed specifically for Vietnamese educational content

## 📞 Support

For questions or issues, please open an issue on the GitHub repository or contact the development team.

---

**Note**: This system is designed specifically for Vietnamese high school textbooks and may require adaptation for other educational content or languages.
