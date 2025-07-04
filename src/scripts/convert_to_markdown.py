import os
import argparse
import logging
from llama_index.readers.llama_parse import LlamaParse
from llama_index.core.node_parser import MarkdownNodeParser

# Simple logger setup
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def pdf_to_markdown(file_path: str, output_path: str):
    """Convert PDF document to markdown

    Args:
        file_path (str): input PDF file path
        output_path (str): output markdown file path
    """
    logger.info("Converting PDF to Markdown...")
    logger.info(f"Input: {file_path}")
    logger.info(f"Output: {output_path}")

    try:
        # Initialize LlamaParse with Vietnamese
        parser = LlamaParse(result_type="markdown", language="vi")

        # Load and parse PDF
        documents = parser.load_data(file_path)

        # Convert to markdown nodes
        node_parser = MarkdownNodeParser()
        nodes = node_parser.get_nodes_from_documents(documents)
        logger.info(f"---Parsed {len(nodes)} markdown node(s)")

        # Join all text
        markdown_content = "\n\n".join(node.text for node in nodes)

        # Save to file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)

        logger.info(f"---Markdown saved to {output_path}")

    except Exception:
        logger.exception("---An error occurred during conversion")


def batch_convert_pdfs(input_folder: str, output_folder: str):
    """
    Convert all PDF files in a specified input folder to Markdown using LlamaParse,
    and save the resulting .md files to the specified output folder.

    This function relies on an existing `pdf_to_markdown(file_path, output_path)` function
    that performs the conversion for a single file.

    Args:
        input_folder (str): Path to the folder containing PDF files to convert.
        output_folder (str): Path to the folder where Markdown files will be saved.

    Notes:
        - Only files ending in '.pdf' (case-insensitive) will be processed.
        - Output Markdown filenames will match the input filenames, with the `.md` extension.
        - Creates the output folder if it does not exist.
        - Logs progress and errors using the standard logging module.
    """
    logger.info(f"Scanning input folder: {input_folder}")

    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".pdf"):
            input_path = os.path.join(input_folder, filename)
            output_filename = os.path.splitext(filename)[0] + ".md"
            output_path = os.path.join(output_folder, output_filename)

            try:
                pdf_to_markdown(input_path, output_path)
            except Exception:
                logger.exception(f"Failed to convert file: {filename}")

    logger.info("Batch conversion completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch convert PDF files to Markdown using LlamaParse (language: Vietnamese)"
    )
    parser.add_argument("input_folder", type=str, help="Folder containing PDF files")
    parser.add_argument("output_folder", type=str, help="Folder to save Markdown files")

    args = parser.parse_args()
    batch_convert_pdfs(args.input_folder, args.output_folder)
