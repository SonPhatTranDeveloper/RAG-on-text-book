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
            "- If the document contains images, figures, charts, diagrams, or other visual elements, "
            "describe them clearly in Vietnamese. Ensure that any caption or inferred description "
            "related to non-text content is also written in Vietnamese.\n"
            "- Do not use HTML tags or any other formatting. "
        ),
        page_separator="\n== {pageNumber} ==\n",
    )
