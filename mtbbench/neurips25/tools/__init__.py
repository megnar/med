"""Optional benchmark tools are imported only when requested."""

def __getattr__(name):
    if name == 'search_pubmed':
        from neurips25.tools.pubmed import search_pubmed
        return search_pubmed
    raise AttributeError(name)
