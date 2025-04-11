
import markdown

def preprocess_md_to_html(md_text: str,model_name: str) -> str:
    """
    Convert a Markdown-like string into HTML.
    Uses Python-Markdown under the hood, so you get support for:
      - ATX headings (#, ##, …)
      - bold (**bold**), italic (*italic*), combined (***both***)
      - inline `code` and fenced code blocks
      - lists, links, images, blockquotes, etc.
    """
    # You can enable extensions as needed:
    extensions = [
        "fenced_code",      # ```code blocks
        "codehilite",       # syntax highlighting (needs Pygments)
        "tables",           # pipe-style tables
        "nl2br",            # newline -> <br>
    ]
    # html = markdown.markdown(f"**{model_name}** : "+md_text, extensions=extensions)
    html = markdown.markdown(md_text, extensions=extensions)

    return html


def check_persian(query:str) ->bool:
    """
    Check if the given string contains any Persian (Farsi) characters.
    Persian characters are typically in the Unicode range \u0600-\u06FF.
    """
    for char in query:
        if '\u0600' <= char <= '\u06FF':
            return True
    return False
