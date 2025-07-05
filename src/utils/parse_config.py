from llama_index.readers.llama_parse import LlamaParse


def get_baseline_llamaparse() -> LlamaParse:
    """
    Create a baseline LlamaParse instance configured for Vietnamese language.

    Returns:
        LlamaParse: Baseline parser instance.
    """
    return LlamaParse(result_type="markdown", language="vi")


def get_lvm_llamaparse() -> LlamaParse:
    """
    Create a layout-aware LlamaParse (LVM-enhanced) instance for Vietnamese,
    using a vision model for structured parsing.

    Returns:
        LlamaParse: Enhanced parser with layout model enabled.
    """
    return LlamaParse(
        result_type="markdown",
        language="vi",
        use_vendor_multimodal_model=True,
        vendor_multimodal_model_name="anthropic-sonnet-3.5",
        system_prompt_append=(
            "This is a book for Vietnamese high-school students, "
            "written in Vietnamese. The text is structured with headings, sections, and other elements. "
            "Your task is to extract the text content in a logical reading order, preserving the structure "
            "and hierarchy such as headings, sections, tables, and bullet points. \n"
            "- When mathematical equations are encountered, transcribe them with high fidelity in LaText format, "
            "maintaining their layout and special characters, and wherever possible, wrap inline equations with `$` and display equations with `$$`. \n"
            "- If the document contains images, figures, charts, diagrams, or other visual elements, "
            "describe them clearly in Vietnamese. Ensure that any caption or inferred description "
            "related to non-text content is also written in Vietnamese.\n"
            "- Do not use HTML tags or any other formatting. "
        ),
        page_separator="\n== {pageNumber} ==\n",
    )


def get_english_book_llamaparse_config() -> LlamaParse:
    """
    Create a layout-aware LlamaParse (LVM-enhanced) instance for a general
    English book.

    Returns:
        LlamaParse: Enhanced parser for English books.
    """
    return LlamaParse(
        result_type="markdown",
        language="en",
        use_vendor_multimodal_model=True,
        vendor_multimodal_model_name="anthropic-sonnet-3.5",
        system_prompt_append=(
            "This is an English book. Your task is to extract the text content "
            "in a logical reading order, preserving the structure and hierarchy "
            "such as headings, sections, paragraphs, tables, and bullet points. \n"
            "- Please ignore any Vietnamese text or content that is not in English. \n"
            "- If the document contains images, figures, charts, diagrams, or other visual elements, "
            "describe them clearly and concisely in English. Ensure that any caption or inferred description "
            "related to non-text content is also written in English.\n"
            "- Do not use HTML tags or any other formatting. "
        ),
        page_separator="\n== {pageNumber} ==\n",
    )
