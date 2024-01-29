import re


def normalize(
    label: str,
    do_lower: bool = True,
    strip_punct: bool = True,
    split_colon: bool = False,
) -> str:
    # Sometimes models wrongfully output a field-prefix, which we can remove.
    if split_colon:
        label = label.split(":")[1] if ":" in label else label

    # Remove leading and trailing newlines
    label = label.strip("\n")

    # Remove leading and trailing punctuation and newlines
    # NOTE: leading and trailing punctuation removal might hurt for e.g. drug and medical reaction ontologies.
    if strip_punct:
        label = re.sub(r"^[^\w\s]+|[^\w\s]+$", "", label, flags=re.UNICODE)

    # Remove leading and trailing newlines
    label = label.strip("\n")

    # NOTE: lowering the labels might hurt for case-sensitive ontologies.
    if do_lower:
        return label.strip().lower()
    else:
        return label.strip()


# given a comma-separated string of labels, parse into a list
def extract_labels_from_string(
    labels: str,
    do_lower: bool = True,
    strip_punct: bool = True,
    split_colon: bool = False,
) -> list[str]:
    return [
        normalize(r, do_lower=do_lower, strip_punct=strip_punct)
        for r in labels.split(",")
    ]


# given a list of comma-separated string of labels, parse into a list
def extract_labels_from_strings(
    labels: list[str],
    do_lower: bool = True,
    strip_punct: bool = True,
    split_colon: bool = False,
) -> list[str]:
    labels = [
        normalize(
            r, do_lower=do_lower, strip_punct=strip_punct, split_colon=split_colon
        )
        for r in labels
    ]
    labels = ", ".join(labels)
    return extract_labels_from_string(
        labels, do_lower=do_lower, strip_punct=strip_punct, split_colon=split_colon
    )
