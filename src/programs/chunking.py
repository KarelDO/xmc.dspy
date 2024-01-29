from .config import IreraConfig


class Chunker:
    def __init__(self, config: IreraConfig):
        self.config = config
        self.chunk_context_window = config.chunk_context_window
        self.chunk_max_windows = config.chunk_max_windows
        self.chunk_window_overlap = config.chunk_window_overlap

    def __call__(self, text):
        snippet_idx = 0

        while snippet_idx < self.chunk_max_windows and text:
            endpos = int(self.chunk_context_window * (1.0 + self.chunk_window_overlap))
            snippet, text = text[:endpos], text[endpos:]

            next_newline_pos = snippet.rfind("\n")
            if (
                text
                and next_newline_pos != -1
                and next_newline_pos >= self.chunk_context_window // 2
            ):
                text = snippet[next_newline_pos + 1 :] + text
                snippet = snippet[:next_newline_pos]

            yield snippet_idx, snippet.strip()
            snippet_idx += 1
