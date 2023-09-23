import textwrap


def wrap_text_preserve_newlines(text: str, width: int = 110) -> str:
    """
    Wraps the input text to the specified width while preserving any existing newline characters.

    Args:
        text (str): The text to wrap.
        width (int, optional): The maximum width of each line. Defaults to 110.

    Returns:
        str: The wrapped text.
    """
    # Split the input text into lines based on newline characters
    lines = text.split('\n')
    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text


def generate_html_response(query: str, response: dict) -> str:
    """
    Generates an HTML response for the given query and response.

    Args:
        query (str): The query that was asked.
        response (dict): The response to the query.

    Returns:
        str: The HTML response.
    """
    text = f'<p><span style="font-weight: bold;">Q:</span> {query}</p>'
    wrapper_text = wrap_text_preserve_newlines(response["result"])
    text += f'<p><span style="font-weight: bold;">A:</span>{wrapper_text}</p>'

    # Add Sources to the text
    text += '<p style="font-weight: bold;">Sources:</p>'
    text += "<ul style='padding-left: 0;'>"
    for source in response["source_documents"]:
        text += f'<li style="list-style-type: none;">{source.metadata.get("source", "")}</li>'
    text += "</ul>"

    # Wrap the HTML content in a div with a chat-response class
    return f'<div style="background-color: rgb(38, 39, 48); padding: 10px; border-radius: 5px;">{text}</div>'