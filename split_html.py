with open("ner-web/src/templates/index.html") as f:
    lines = f.readlines()

start_idx = 0
end_idx = 0
for i, line in enumerate(lines):
    if '<main class="main">' in line:
        start_idx = i
    if '</main>' in line:
        end_idx = i
        break

base_top = lines[:start_idx]
ner_content = lines[start_idx:end_idx+1]
base_bottom = lines[end_idx+1:]

with open("ner-web/templates/base.html", "w") as f:
    f.writelines(base_top)
    f.write('    {% block content %}{% endblock %}\n')
    f.writelines(base_bottom)

with open("ner-web/templates/ner.html", "w") as f:
    f.write('{% if !is_fragment %}{% extends "base.html" %}{% endif %}\n')
    f.write('{% block content %}\n')
    f.writelines(ner_content)
    f.write('\n{% endblock %}\n')

