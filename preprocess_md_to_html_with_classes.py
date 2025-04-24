import markdown
from bs4 import BeautifulSoup

def preprocess_md_to_html_with_classes(md_text: str) -> str:
    """
    Convert Markdown to HTML and add classes to certain elements.
    """
    # Convert Markdown to HTML
    extensions = [
        "fenced_code",
        "codehilite",
        "tables",
        "nl2br",
    ]
    html = markdown.markdown(md_text, extensions=extensions)

    # Parse HTML and add custom classes
    soup = BeautifulSoup(html, "html.parser")

    # Example: add classes to specific tags
    tag_class_map = {
        "h1": "heading heading-1",
        "h2": "heading heading-2",
        "p": "paragraph",
        "ul": "list list-unordered",
        "ol": "list list-ordered",
        "code": "inline-code",
        "pre": "code-block",
        "table": "table table-striped",
        "blockquote": "blockquote",
    }

    for tag, class_name in tag_class_map.items():
        for elem in soup.find_all(tag):
            existing_classes = elem.get("class", [])
            new_classes = class_name.split()
            elem["class"] = list(set(existing_classes + new_classes))

    return str(soup)
