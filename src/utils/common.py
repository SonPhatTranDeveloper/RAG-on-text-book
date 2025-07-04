import re


def extract_metadata_from_filename(file_name: str) -> dict:
    """
    Extract metadata from a filename formatted as '[grade]_[book_type]_[book_subject].pdf'
    using regular expressions.

    Args:
        file_name (str): The filename to extract metadata from.

    Returns:
        dict: A dictionary containing the extracted metadata.
    """
    # Define the pattern for the filename
    pattern = (
        r"^(grade_\d+)_((?:chan_troi_sang_tao|ket_noi_tri_thuc|canh_dieu))_(.*)\.pdf$"
    )

    match = re.match(pattern, file_name)

    if match:
        grade = match.group(1)
        book_type = match.group(2)
        book_subject = match.group(3)

        return {"grade": grade, "book_type": book_type, "book_subject": book_subject}
    else:
        # If the filename does not match the expected pattern
        return {
            "book_grade": None,
            "book_type": None,
            "book_subject": file_name.replace(
                ".pdf", ""
            ),  # Return the base filename if no match
        }
