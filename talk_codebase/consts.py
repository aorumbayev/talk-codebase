from langchain.document_loaders import CSVLoader, UnstructuredWordDocumentLoader, UnstructuredEPubLoader, \
    PDFMinerLoader, UnstructuredMarkdownLoader, TextLoader

EXCLUDE_DIRS = ['__pycache__', '.venv', '.git', '.idea', 'venv', 'env', 'node_modules', 'dist', 'build', '.vscode',
                '.github', '.gitlab']
ALLOW_FILES = ['.txt', '.js', '.mjs', '.ts', '.tsx', '.css', '.scss', '.less', '.html', '.htm', '.json', '.py',
               '.java', '.c', '.cpp', '.cs', '.go', '.php', '.rb', '.rs', '.swift', '.kt', '.scala', '.m', '.h',
               '.sh', '.pl', '.pm', '.lua', '.sql']
EXCLUDE_FILES = ['requirements.txt', 'package.json', 'package-lock.json', 'yarn.lock']
MODEL_TYPES = {
    "OPENAI": "openai",
    "LOCAL": "local",
}
DEFAULT_CONFIG = {
    "max_tokens": "1048",
    "chunk_size": "500",
    "chunk_overlap": "50",
    "k": "4",
    "model_name": "gpt-3.5-turbo",
    "model_path": "models/ggml-gpt4all-l13b-snoozy.bin",
    "model_type": MODEL_TYPES["OPENAI"],
}

LOADER_MAPPING = {
    ".csv": {
        "loader": CSVLoader,
        "args": {}
    },
    ".doc": {
        "loader": UnstructuredWordDocumentLoader,
        "args": {}
    },
    ".docx": {
        "loader": UnstructuredWordDocumentLoader,
        "args": {}
    },
    ".epub": {
        "loader": UnstructuredEPubLoader,
        "args": {}
    },
    ".md": {
        "loader": UnstructuredMarkdownLoader,
        "args": {}
    },
    ".pdf": {
        "loader": PDFMinerLoader,
        "args": {}
    }
}

for ext in ALLOW_FILES:
    if ext not in LOADER_MAPPING:
        LOADER_MAPPING[ext] = {
            "loader": TextLoader,
            "args": {
                "encoding": "utf8"
            }
        }
