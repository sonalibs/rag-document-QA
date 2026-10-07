def chunk_text(text, chunk_size=800, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))      # NEW: stop at end of text
        if end < len(text):                           # NEW
            space = text.rfind(" ", start, end)       # NEW: last space before the cut
            if space > start:                         # NEW
                end = space                           # NEW: cut at the space, not mid-word
        chunks.append(text[start:end].strip())        # CHANGED: .strip()
        if end == len(text):                          # NEW
            break                                     # NEW: stop, no repeated tail chunks
        start = end - overlap
        space = text.find(" ", start)                 # NEW: next space after stepping back
        if space != -1:                               # NEW
            start = space + 1                         # NEW: start at a word beginning
    return chunks