from pathlib import Path

import pandas as pd


def before_body(path_latex: Path, alpha_name_escaped: str):
    with open(path_latex, "w", encoding="utf-8") as f:
        # LaTeX document header
        f.write(
            r"""
\documentclass[12pt,a4paper]{article}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{geometry}
\usepackage{graphicx}
\usepackage{subcaption}
\usepackage{fancyhdr}
\usepackage{titlesec}
\usepackage{color}
\usepackage{xcolor}
\usepackage{hyperref}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{multirow}
\usepackage{wrapfig}
\usepackage{float}
\usepackage{colortbl}
\usepackage{pdflscape}
\usepackage{tabu}
\usepackage{threeparttable}
\usepackage{threeparttablex}
\usepackage{makecell}

% Page geometry
\geometry{
    left=2.5cm,
    right=2.5cm,
    top=2.5cm,
    bottom=2.5cm,
    headheight=15pt
}

% Custom colors
\definecolor{sectioncolor}{RGB}{0,122,204}
\definecolor{subsectioncolor}{RGB}{44,62,80}
\definecolor{subsubsectioncolor}{RGB}{127,140,141}

% Title formatting
\titleformat{\section}
    {\Large\bfseries\color{sectioncolor}}
    {\thesection}
    {1em}
    {}

\titleformat{\subsection}
    {\large\bfseries\color{subsectioncolor}}
    {\thesubsection}
    {1em}
    {}

\titleformat{\subsubsection}
    {\normalsize\bfseries\color{subsubsectioncolor}}
    {\thesubsubsection}
    {1em}
    {}

% Header and footer
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\textcolor{subsubsectioncolor}{\textit{Performance Analysis Report}}}
\fancyhead[R]{\textcolor{subsubsectioncolor}{\textit{Alpha: """
            + alpha_name_escaped
            + r"""}}}
\fancyfoot[C]{\thepage}
\renewcommand{\headrulewidth}{0.4pt}
\renewcommand{\footrulewidth}{0.4pt}

% Disable all hyperref coloring for table of contents
\hypersetup{allcolors=black}

% Remove all borders and boxes from hyperref links
\hypersetup{pdfborder={0 0 0},linkbordercolor=white,urlbordercolor=white,citebordercolor=white,filebordercolor=white,menubordercolor=white,runbordercolor=white}

% Disable link borders completely
\hypersetup{pdfborderstyle={},linkbordercolor={},urlbordercolor={},citebordercolor={},filebordercolor={},menubordercolor={},runbordercolor={}}

% Document info
\title{\Huge\textbf{Performance Analysis Report}\\[0.5cm]\Large\textcolor{sectioncolor}{"""
            + alpha_name_escaped
            + r"""}}
\author{Backtest System}
\date{\today}

\begin{document}

\maketitle

\newpage

% Disable hyperref styling for table of contents
\hypersetup{linktoc=all,linktocpage=all,bookmarks=false}

\tableofcontents
\newpage
"""
        )


def part(path_latex: Path, text: str):
    with open(path_latex, "a", encoding="utf-8") as f:
        # LaTeX document header
        f.write(
            rf"""
\part{{{text}}}
"""
        )


def section(path_latex: Path, text: str):
    with open(path_latex, "a", encoding="utf-8") as f:
        # LaTeX document header
        f.write(
            rf"""
\section{{{text}}}
"""
        )


def subsection(path_latex: Path, text: str):
    with open(path_latex, "a", encoding="utf-8") as f:
        # LaTeX document header
        f.write(
            rf"""
\subsection{{{text}}}
"""
        )


def subsubsection(path_latex: Path, text: str):
    with open(path_latex, "a", encoding="utf-8") as f:
        # LaTeX document header
        f.write(
            rf"""
\subsubsection{{{text}}}
"""
        )


def includegraph(path_latex: Path, path_graph: Path, caption: str):

    path_graph = str(path_graph)

    with open(path_latex, "a", encoding="utf-8") as f:
        f.write(
            f"""
\\begin{{figure}}[H]
\\centering
\\includegraphics[width=\\textwidth]{{{path_graph}}}
\\caption{{{caption}}}
\\end{{figure}}
"""
        )


def include_ic_metrics_table(path_latex: Path, df_metrics: pd.DataFrame, caption: str):

    with open(path_latex, "a", encoding="utf-8") as f:
        f.write(
            f"""
\\begin{{table}}[H]
\\centering
\\begin{{tabular}}{{lc}}
\\hline
\\hline
\\textbf{{Metric}} & \\textbf{{Value}} \\\\
\\hline
"""
        )

        # Add each row of data
        for _, row in df_metrics.iterrows():
            metric_escaped = row["metrics"].replace("_", r"\_")
            value = row["value"]

            f.write(f"{metric_escaped} & {value} \\\\\n")

        f.write(
            f"""
\\hline
\\end{{tabular}}
\\caption{{{caption}}}
\\end{{table}}
"""
        )


def include_multiple_plots_grid(
    path_latex: Path, plot_paths: list, caption: str, n_rows: int = 2, n_cols: int = 2
):
    """
    Include multiple plots in a grid layout in LaTeX

    :param path_latex: Path to LaTeX file
    :param plot_paths: List of plot file paths
    :param caption: Main caption for the grid
    :param n_rows: Number of rows in the grid
    :param n_cols: Number of columns in the grid
    """
    with open(path_latex, "a", encoding="utf-8") as f:
        f.write(
            f"""
\\begin{{figure}}[H]
\\centering
\\begin{{subfigure}}[b]{{0.48\\textwidth}}
    \\includegraphics[width=\\textwidth]{{{plot_paths[0]}}}
    \\caption{{Original Distribution}}
\\end{{subfigure}}
\\hfill
\\begin{{subfigure}}[b]{{0.48\\textwidth}}
    \\includegraphics[width=\\textwidth]{{{plot_paths[1]}}}
    \\caption{{[0.01, 0.99] Winsorized}}
\\end{{subfigure}}

\\vspace{{0.5cm}}

\\begin{{subfigure}}[b]{{0.48\\textwidth}}
    \\includegraphics[width=\\textwidth]{{{plot_paths[2]}}}
    \\caption{{[0.03, 0.97] Winsorized}}
\\end{{subfigure}}
\\hfill
\\begin{{subfigure}}[b]{{0.48\\textwidth}}
    \\includegraphics[width=\\textwidth]{{{plot_paths[3]}}}
    \\caption{{[0.05, 0.95] Winsorized}}
\\end{{subfigure}}

\\caption{{{caption}}}
\\end{{figure}}
"""
        )


def include_longshort_metrics_table(
    path_latex: Path, df_metrics: pd.DataFrame, caption: str
):

    with open(path_latex, "a", encoding="utf-8") as f:
        f.write(
            f"""
\\begin{{table}}[H]
\\scriptsize
\\centering
\\begin{{tabular}}{{c|ccccccc}}
\\hline
\\hline
\\textbf{{Mode}} & \\textbf{{Start Period}} & \\textbf{{End Period}} & \\textbf{{CAGR\\%}} & \\textbf{{Sharpe}} & \\textbf{{Max Drawdown}} & \\textbf{{Turnover}} \\\\
\\hline
"""
        )

        # Add each row of data
        for _, row in df_metrics.iterrows():
            mode_escaped = row["mode"].replace("_", r"\_")
            start_period = row["Start Period"]
            end_period = row["End Period"]
            cagr = (
                row["CAGR﹪"].replace("%", r"\%")
                if isinstance(row["CAGR﹪"], str)
                else str(row["CAGR﹪"])
            )
            sharpe = str(row["Sharpe"])
            max_drawdown = (
                row["Max Drawdown"].replace("%", r"\%")
                if isinstance(row["Max Drawdown"], str)
                else str(row["Max Drawdown"])
            )
            turnover = str(row["Turnover"])

            f.write(
                f"{mode_escaped} & {start_period} & {end_period} & {cagr} & {sharpe} & {max_drawdown} & {turnover} \\\\\n"
            )

        f.write(
            f"""
\\hline
\\end{{tabular}}
\\caption{{{caption}}}
\\end{{table}}
"""
        )


def include_stratify_metrics_table(
    path_latex: Path, df_metrics: pd.DataFrame, caption: str
):
    """
    Include stratify metrics table in LaTeX

    :param path_latex: Path to LaTeX file
    :param df_metrics: DataFrame with stratify metrics (should have 'metric' and 'value' columns)
    :param caption: Caption for the table
    """
    with open(path_latex, "a", encoding="utf-8") as f:
        f.write(
            f"""
\\begin{{table}}[H]
\\scriptsize
\\centering
\\begin{{tabular}}{{c|ccccccc}}
\\hline
\\hline
\\textbf{{Layer}} & \\textbf{{Start Period}} & \\textbf{{End Period}} & \\textbf{{CAGR\\%}} & \\textbf{{Sharpe}} & \\textbf{{Max Drawdown}} & \\textbf{{Turnover}} \\\\
\\hline
"""
        )

        # Add each row of data
        for _, row in df_metrics.iterrows():
            layer_escaped = row["layer"].replace("_", r"\_")
            start_period = row["Start Period"]
            end_period = row["End Period"]
            cagr = (
                row["CAGR﹪"].replace("%", r"\%")
                if isinstance(row["CAGR﹪"], str)
                else str(row["CAGR﹪"])
            )
            sharpe = str(row["Sharpe"])
            max_drawdown = (
                row["Max Drawdown"].replace("%", r"\%")
                if isinstance(row["Max Drawdown"], str)
                else str(row["Max Drawdown"])
            )
            turnover = str(row["Turnover"])

            f.write(
                f"{layer_escaped} & {start_period} & {end_period} & {cagr} & {sharpe} & {max_drawdown} & {turnover} \\\\\n"
            )

        f.write(
            f"""
\\hline
\\end{{tabular}}
\\caption{{{caption}}}
\\end{{table}}
"""
        )


def newpage(path_latex: Path):
    with open(path_latex, "a", encoding="utf-8") as f:
        # LaTeX document header
        f.write(
            rf"""
\newpage
"""
        )


def end(path_latex: Path):
    with open(path_latex, "a", encoding="utf-8") as f:
        f.write(
            r"""
\end{document}
"""
        )
