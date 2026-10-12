"""Build the School of Computer Science thesis-template version from manuscript/main.tex.

The manuscript (manuscript/main.tex, article class) stays the single source of content. This script
wraps its text in the School's usydthesis.cls template: title page, Student Plagiarism Compliance
Statement, abstract, acknowledgements, contents, list of figures and list of tables, then the
sections as chapters, the AI-use declaration, references and appendices. Re-run it after every
content change instead of editing the generated file by hand.

Usage:
  python build_thesis.py <main.tex> <template_dir> <out_dir>
      [--sid SID] [--supervisor NAME] [--degree DEGREE] [--acknowledgements FILE]
      [--signature NAME] [--sign-date DATE]

--signature and --sign-date type the signature (in italics) and the date into the Plagiarism
Compliance Statement; without them those two cells stay blank for a handwritten signature.
--degree is empty by default, as in the School's LaTeX and Word templates and the Research Pathway
sample thesis, so the title page ends with "the requirements for the Research Pathway Project".

Layout follows the template (title page, compliance statement, abstract, acknowledgements, contents,
lists of figures and tables, chapters, bibliography, appendices with contents depth 1). Two
presentation settings are added: \raggedbottom, so short pages before large tables and figures
leave space at the foot instead of stretching the gaps between paragraphs, and list-of-figures and
list-of-tables entries cut at the first colon of long captions.

<template_dir> is the unpacked thesis.zip from Canvas (it holds usydthesis.cls and style/).
<out_dir> receives thesis.tex, the class and style files, references.bib and figures/ (copied from
the manuscript folder). Compile with: pdflatex thesis; biber thesis; pdflatex thesis; pdflatex thesis
"""
import argparse
import re
import shutil
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("main", type=Path)
ap.add_argument("template", type=Path)
ap.add_argument("out", type=Path)
ap.add_argument("--sid", default="[SID]")
ap.add_argument("--supervisor", default="[Supervisor]")
ap.add_argument("--degree", default="")
ap.add_argument("--acknowledgements", type=Path)
ap.add_argument("--signature", help="typed signature for the Plagiarism Compliance Statement")
ap.add_argument("--sign-date", help="date next to the signature, e.g. '12 October 2026'")
args = ap.parse_args()

src = args.main.read_text(encoding="utf-8").replace("\r\n", "\n")


def between(text, start, end):
    i = text.index(start)
    j = text.index(end, i + len(start))
    return text[i + len(start):j]


title = re.sub(r"\s+", " ", between(src, r"\title{", "}\n\\author")).strip()
author = between(src, r"\author{", r"\\").strip()
abstract = between(src, "\\addcontentsline{toc}{section}{Abstract}\n", "\\section{Introduction}").strip()
body = src[src.index("\\section{Introduction}"):src.index("\\section*{Declaration of Generative AI Use}")]
declaration = between(src, "\\addcontentsline{toc}{section}{Declaration of Generative AI Use}\n",
                      "\\appendix").strip()
appendix = between(src, "\\appendix\n", "\\clearpage\n\\printbibliography").strip()
assert body.count("\\appendix") == 0 and "\\printbibliography" not in appendix

LEVEL = {"section": "chapter", "subsection": "section", "subsubsection": "subsection"}


def promote(text):
    return re.sub(r"\\(subsubsection|subsection|section)(\*?)\{",
                  lambda m: "\\" + LEVEL[m.group(1)] + m.group(2) + "{", text)


# labels of top-level sections: in the body they become chapters ("Chapter~\ref"), in the appendix
# they become appendices ("Appendix~\ref")
HEADING_LABEL = r"\\section\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}\s*\\label\{([^}]+)\}"
chapter_labels = set(re.findall(HEADING_LABEL, body))
appendix_labels = set(re.findall(HEADING_LABEL, appendix))
body, appendix = promote(body), promote(appendix)


def rename_refs(text):
    for labels, name in ((chapter_labels, "Chapter"), (appendix_labels, "Appendix")):
        for label in labels:
            text = text.replace(f"Sections~\\ref{{{label}}}", f"{name}s~\\ref{{{label}}}")
            text = text.replace(f"Section~\\ref{{{label}}}", f"{name}~\\ref{{{label}}}")
    return text


body, appendix = rename_refs(body), rename_refs(appendix)
abstract, declaration = rename_refs(abstract), rename_refs(declaration)


def top_level_colons(s):
    """Positions of ': ' outside braces and inline maths."""
    depth, maths, found, i = 0, False, [], 0
    while i < len(s):
        c = s[i]
        if c == "\\":
            if s.startswith("\\(", i) or s.startswith("\\)", i):
                maths = s[i + 1] == "("
            i += 2
            continue
        if c == "$":
            maths = not maths
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        elif c == ":" and depth == 0 and not maths and s[i + 1:i + 2] == " ":
            found.append(i)
        i += 1
    return found


def short_captions(text):
    """Give every \\caption{...} a short list-of-figures/tables entry: its first sentence, cut at
    the first colon when that sentence is long (over 100 characters)."""
    out, i = [], 0
    while True:
        j = text.find("\\caption{", i)
        if j < 0:
            return "".join(out) + text[i:]
        k, depth = j + len("\\caption{"), 1
        while depth:
            depth += {"{": 1, "}": -1}.get(text[k], 0)
            k += 1
        full = text[j + len("\\caption{"):k - 1]
        flat = re.sub(r"\s+", " ", full).strip()
        cut = re.search(r"(?<!\be\.g)(?<!\bi\.e)(?<!\bvs)\.(?:\s|$)", flat)
        short = flat[:cut.start() + 1] if cut else flat
        colons = top_level_colons(short)
        if len(short) > 100 and colons and len(short[:colons[0]].split()) >= 2:
            short = short[:colons[0]].rstrip() + "."
        balanced = short.count("{") == short.count("}") and short.count("$") % 2 == 0 \
            and short.count("\\(") == short.count("\\)")
        caption = f"\\caption[{{{short}}}]{{{full}}}" if short != flat and balanced else text[j:k]
        out.append(text[i:j] + caption)
        i = k


body, appendix = short_captions(body), short_captions(appendix)

acknowledgements = (args.acknowledgements.read_text(encoding="utf-8").strip() if args.acknowledgements else
                    "I thank my supervisor, " + args.supervisor + ", for guidance throughout this project.")

plagiarism = (args.template / "plagiarism" / "plagiarism.tex").read_text(encoding="utf-8").strip()


def tex_escape(text):
    return re.sub(r"([&%$#_{}])", r"\\\1", text)


if args.signature or args.sign_date:
    # the template's two \hspace{0.4\textwidth} cells are removed by the tabular's \unskip, so the
    # filled row uses a 0.3\textwidth box for the signature (two 0.4 boxes would overrun the line)
    blank_row = (r"\textbf{Signature}: & \hspace{0.4\textwidth} & \textbf{Date}: & "
                 r"\hspace{0.4\textwidth}\\[2cm]")
    assert plagiarism.count(blank_row) == 1, "signature row not found in plagiarism.tex"
    signature = r"\textit{" + tex_escape(args.signature) + "}" if args.signature else ""
    sign_date = tex_escape(args.sign_date) if args.sign_date else ""
    plagiarism = plagiarism.replace(
        blank_row,
        r"\textbf{Signature}: & \makebox[0.3\textwidth][l]{" + signature + r"} & \textbf{Date}: & "
        + sign_date + r"\\[2cm]")

preamble = r"""\documentclass{usydthesis}

% Configuration (School of Computer Science thesis template)
\def\degree{@@DEGREE@@}
\def\department{School of Computer Science}
\title{{\bf\Huge @@TITLE@@}}
\author{@@AUTHOR@@}
\def\sid{@@SID@@}
\def\supervisor{@@SUPERVISOR@@}

% Packages: template set reduced to what the manuscript uses, plus the manuscript's own
\usepackage[utf8]{inputenc}
\usepackage[english]{babel}
\usepackage{csquotes}
\usepackage{booktabs,longtable,array,calc,adjustbox,multirow,tabularx}
\usepackage{enumitem}
\usepackage{xurl}
\usepackage[section]{placeins}
\usepackage{float}
\floatplacement{figure}{H}
\usepackage{etoolbox}
\BeforeBeginEnvironment{longtable}{\FloatBarrier}
\usepackage{setspace}
\usepackage{xcolor}
\usepackage[backend=biber,style=apa,sorting=nyt]{biblatex}
\addbibresource{references.bib}
\AtBeginBibliography{\raggedright}
\usepackage[colorlinks=false,plainpages=false,pdfborder={0 0 0},backref=false,unicode]{hyperref}
\hypersetup{pdftitle={@@TITLE@@},pdfauthor={@@AUTHOR@@}}
\usepackage[all]{hypcap}
\setlength{\emergencystretch}{3em}
\setlist{itemsep=3pt,topsep=4pt}
\newcommand{\tightlist}{\setlength{\itemsep}{0pt}\setlength{\parskip}{0pt}}
\newcommand{\pandocbounded}[1]{\begin{adjustbox}{max size={\linewidth}{0.76\textheight}}#1\end{adjustbox}}
\providecommand{\passthrough}[1]{#1}

% Page size, as in the template
\oddsidemargin=0cm
\evensidemargin=0cm
\textwidth=6.2677165in
\renewcommand{\thepage}{\roman{page}}
\raggedbottom

\begin{document}
\maketitle
\setstretch{1.5}

\cleardoublepage
\phantomsection
@@PLAGIARISM@@

\cleardoublepage
\phantomsection
\chapter*{Abstract}
@@ABSTRACT@@

\cleardoublepage
\phantomsection
\chapter*{Acknowledgements}
@@ACKNOWLEDGEMENTS@@

\cleardoublepage
\setcounter{tocdepth}{2}
\tableofcontents
{\makeatletter
\renewcommand*\numberline[1]{\hb@xt@\@tempdima{#1 \hfil}\hspace*{1em}}
\makeatother
\listoffigures
\listoftables
\cleardoublepage
}

\setcounter{page}{1}
\setcounter{chapter}{0}
\renewcommand{\thepage}{\arabic{page}}
\setupParagraphs

"""

front = preamble
for key, value in {"DEGREE": args.degree, "TITLE": title, "AUTHOR": author, "SID": args.sid,
                   "SUPERVISOR": args.supervisor, "PLAGIARISM": plagiarism, "ABSTRACT": abstract,
                   "ACKNOWLEDGEMENTS": acknowledgements}.items():
    front = front.replace(f"@@{key}@@", value)
assert "@@" not in front
document = (front
            + body.rstrip() + "\n\n"
            + "\\chapter*{Declaration of Generative AI Use}\n"   # amsbook lists starred chapters itself
            + declaration + "\n\n"
            + "{\\setstretch{1.25}\n\\cleardoublepage\n\\phantomsection\n"
            + "\\printbibliography[heading=bibliography,title={Bibliography}]\n}\n\n"
            + "\\appendix\n\\addtocontents{toc}{\\protect\\setcounter{tocdepth}{1}}\n"
            + appendix + "\n\n\\end{document}\n")

args.out.mkdir(parents=True, exist_ok=True)
(args.out / "thesis.tex").write_text(document, encoding="utf-8")
shutil.copy2(args.template / "usydthesis.cls", args.out / "usydthesis.cls")
shutil.copytree(args.template / "style", args.out / "style", dirs_exist_ok=True)
shutil.copy2(args.main.parent / "references.bib", args.out / "references.bib")
shutil.copytree(args.main.parent / "figures", args.out / "figures", dirs_exist_ok=True)
print(f"wrote {args.out / 'thesis.tex'}: {len(chapter_labels)} chapter labels, "
      f"{body.count(chr(92) + 'chapter{')} chapters in the body")
