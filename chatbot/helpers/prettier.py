import os


def prettify_source(source):
    raw_document = source.get("document") or ""
    document = os.path.basename(raw_document) if raw_document else "Unknown"
    score = source.get("score")
    content_preview = source.get("content_preview")
    return f"• **{document}** with score ({round(score,2)}) \n\n **Preview:** \n {content_preview} \n"
