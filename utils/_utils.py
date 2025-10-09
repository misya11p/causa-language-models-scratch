import re
import ast


def decode_text(instance):
    actual_bytes = ast.literal_eval(instance["text"])
    decoded_string = actual_bytes.decode('utf-8')
    paragraphs = re.findall(
        r"_START_PARAGRAPH_\n(.*?)(?=\n_START_PARAGRAPH_|\Z)",
        decoded_string,
        re.DOTALL
    )
    instance["text"] = "\n".join(paragraphs)
    return instance
