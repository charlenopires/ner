import os
import re

def process():
    with open("ner-web/src/templates/index.html") as f:
        text = f.read()

    # Modify links to use full page hx-select approach
    text = text.replace('href="#" hx-get="/" hx-target="#main-content"', 'href="/" hx-get="/" hx-target="#main-content" hx-select="#main-content" hx-push-url="true"')
    text = text.replace('href="#" hx-get="/fragment/tokenizer" hx-target="#main-content"', 'href="/tokenizer" hx-get="/tokenizer" hx-target="#main-content" hx-select="#main-content" hx-push-url="true"')
    text = text.replace('href="#" hx-get="/fragment/ned" hx-target="#main-content"', 'href="/ned" hx-get="/ned" hx-target="#main-content" hx-select="#main-content" hx-push-url="true"')
    text = text.replace('href="#" hx-get="/fragment/nel" hx-target="#main-content"', 'href="/nel" hx-get="/nel" hx-target="#main-content" hx-select="#main-content" hx-push-url="true"')
    text = text.replace('href="#" hx-get="/fragment/sota" hx-target="#main-content"', 'href="/sota" hx-get="/sota" hx-target="#main-content" hx-select="#main-content" hx-push-url="true"')

    lines = text.split('\n')
    lines = [l + '\n' for l in lines] # restore newlines

    start_idx = 0
    end_idx = 0
    for i, line in enumerate(lines):
        if '<div id="main-content">' in line:
            start_idx = i + 1
        if '  </div> <!-- main-content -->' in line:
            end_idx = i - 1
            break

    base_top = lines[:start_idx]
    ner_content = lines[start_idx:end_idx+1]
    base_bottom = lines[end_idx+1:]

    for i in range(len(base_top)):
        if '-webkit-background-clip: text;' in base_top[i]:
            base_top.insert(i+1, base_top[i].replace('-webkit-', ''))

    os.makedirs("ner-web/templates", exist_ok=True)

    with open("ner-web/templates/base.html", "w") as f:
        for line in base_top:
            f.write(line)
        f.write('    {% block content %}{% endblock %}\n')
        for line in base_bottom:
            f.write(line)

    with open("ner-web/templates/ner.html", "w") as f:
        f.write('{% extends "base.html" %}\n')
        f.write('{% block content %}\n')
        for line in ner_content:
            f.write(line)
        f.write('{% endblock %}\n')

process()
