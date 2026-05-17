from __future__ import annotations

import argparse
import html
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports" / "test-report"


@dataclass(frozen=True)
class TestCase:
    classname: str
    name: str
    time: float
    status: str
    message: str = ""


@dataclass(frozen=True)
class TestSummary:
    tests: int = 0
    failures: int = 0
    errors: int = 0
    skipped: int = 0
    time: float = 0.0
    cases: tuple[TestCase, ...] = ()

    @property
    def passed(self) -> int:
        return self.tests - self.failures - self.errors - self.skipped

    @property
    def pass_rate(self) -> float:
        return percentage(self.passed, self.tests)


@dataclass(frozen=True)
class CoverageFile:
    path: str
    line_rate: float
    branch_rate: float
    lines_covered: int
    lines_valid: int


@dataclass(frozen=True)
class CoverageSummary:
    line_rate: float = 0.0
    branch_rate: float = 0.0
    lines_covered: int = 0
    lines_valid: int = 0
    files: tuple[CoverageFile, ...] = ()

    @property
    def line_percent(self) -> float:
        return self.line_rate * 100

    @property
    def branch_percent(self) -> float:
        return self.branch_rate * 100


def main() -> int:
    args = parse_args()
    report_dir = Path(args.report_dir).resolve()
    report_dir.mkdir(parents=True, exist_ok=True)

    junit_path = report_dir / "junit.xml"
    coverage_path = report_dir / "coverage.xml"
    html_path = report_dir / "index.html"

    exit_code = 0
    if not args.no_run:
        exit_code = run_pytest(junit_path, coverage_path)

    test_summary = parse_junit(junit_path)
    coverage_summary = parse_coverage(coverage_path)
    html_path.write_text(
        render_html(test_summary, coverage_summary, exit_code),
        encoding="utf-8",
    )

    print(f"Test report written to: {html_path}")
    return exit_code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a standalone HTML test report from pytest results."
    )
    parser.add_argument(
        "--report-dir",
        default=str(DEFAULT_REPORT_DIR),
        help="Directory for JUnit XML, coverage XML, and the HTML report.",
    )
    parser.add_argument(
        "--no-run",
        action="store_true",
        help="Render the HTML report from existing XML files without running pytest.",
    )
    return parser.parse_args()


def run_pytest(junit_path: Path, coverage_path: Path) -> int:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "tests",
        "--override-ini",
        "addopts=",
        f"--junitxml={junit_path}",
        "--cov=src",
        "--cov-report=term-missing",
        f"--cov-report=xml:{coverage_path}",
        "--cov-fail-under=0",
    ]
    return subprocess.run(command, cwd=PROJECT_ROOT, check=False).returncode


def parse_junit(junit_path: Path) -> TestSummary:
    if not junit_path.exists():
        return TestSummary()

    root = ET.parse(junit_path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    tests = sum(int(suite.attrib.get("tests", 0)) for suite in suites)
    failures = sum(int(suite.attrib.get("failures", 0)) for suite in suites)
    errors = sum(int(suite.attrib.get("errors", 0)) for suite in suites)
    skipped = sum(int(suite.attrib.get("skipped", 0)) for suite in suites)
    duration = sum(float(suite.attrib.get("time", 0.0)) for suite in suites)

    cases: list[TestCase] = []
    for case in root.iter("testcase"):
        status = "passed"
        message = ""
        for child_status in ("failure", "error", "skipped"):
            child = case.find(child_status)
            if child is not None:
                status = child_status
                message = child.attrib.get("message", "") or (child.text or "")
                break

        cases.append(
            TestCase(
                classname=case.attrib.get("classname", ""),
                name=case.attrib.get("name", ""),
                time=float(case.attrib.get("time", 0.0)),
                status=status,
                message=message.strip(),
            )
        )

    return TestSummary(
        tests=tests,
        failures=failures,
        errors=errors,
        skipped=skipped,
        time=duration,
        cases=tuple(cases),
    )


def parse_coverage(coverage_path: Path) -> CoverageSummary:
    if not coverage_path.exists():
        return CoverageSummary()

    root = ET.parse(coverage_path).getroot()
    files: list[CoverageFile] = []

    for class_node in root.findall(".//class"):
        lines = class_node.findall("./lines/line")
        lines_valid = len(lines)
        lines_covered = sum(1 for line in lines if int(line.attrib.get("hits", 0)) > 0)
        files.append(
            CoverageFile(
                path=class_node.attrib.get("filename", ""),
                line_rate=float(class_node.attrib.get("line-rate", 0.0)),
                branch_rate=float(class_node.attrib.get("branch-rate", 0.0)),
                lines_covered=lines_covered,
                lines_valid=lines_valid,
            )
        )

    return CoverageSummary(
        line_rate=float(root.attrib.get("line-rate", 0.0)),
        branch_rate=float(root.attrib.get("branch-rate", 0.0)),
        lines_covered=int(root.attrib.get("lines-covered", 0)),
        lines_valid=int(root.attrib.get("lines-valid", 0)),
        files=tuple(sorted(files, key=lambda item: item.line_rate)),
    )


def render_html(
    test_summary: TestSummary,
    coverage_summary: CoverageSummary,
    exit_code: int,
) -> str:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status_text = "Passing" if exit_code == 0 else "Needs attention"
    status_class = "ok" if exit_code == 0 else "bad"

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>langchain-demo test report</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #17202a;
      --muted: #667085;
      --line: #d8dee8;
      --green: #238b45;
      --red: #c2410c;
      --amber: #b7791f;
      --blue: #2563eb;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Arial, Helvetica, sans-serif;
      line-height: 1.45;
    }}
    header {{
      padding: 28px 32px 18px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }}
    main {{ padding: 24px 32px 40px; }}
    h1 {{ margin: 0; font-size: 28px; font-weight: 700; }}
    h2 {{ margin: 0 0 14px; font-size: 18px; }}
    .meta {{ margin-top: 6px; color: var(--muted); }}
    .status {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      margin-top: 14px;
      padding: 6px 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      font-weight: 700;
    }}
    .status.ok {{ color: var(--green); }}
    .status.bad {{ color: var(--red); }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(140px, 1fr));
      gap: 14px;
      margin-bottom: 22px;
    }}
    .metric, section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    .metric {{ padding: 16px; }}
    .metric .label {{ color: var(--muted); font-size: 13px; }}
    .metric .value {{ margin-top: 6px; font-size: 26px; font-weight: 700; }}
    section {{ padding: 18px; margin-top: 18px; }}
    .bar {{
      height: 12px;
      overflow: hidden;
      background: #e7ebf2;
      border-radius: 999px;
    }}
    .bar span {{
      display: block;
      height: 100%;
      width: var(--value);
      background: var(--color);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      padding: 10px 8px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }}
    th {{ color: var(--muted); font-size: 12px; text-transform: uppercase; }}
    .right {{ text-align: right; }}
    .status-text.passed {{ color: var(--green); }}
    .status-text.failure, .status-text.error {{ color: var(--red); }}
    .status-text.skipped {{ color: var(--amber); }}
    .path {{ font-family: Consolas, Monaco, monospace; }}
    @media (max-width: 820px) {{
      header, main {{ padding-left: 16px; padding-right: 16px; }}
      .grid {{ grid-template-columns: repeat(2, minmax(120px, 1fr)); }}
      table {{ font-size: 13px; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>langchain-demo test report</h1>
    <div class="meta">Generated {escape(generated_at)}</div>
    <div class="status {status_class}">{escape(status_text)}</div>
  </header>
  <main>
    {render_metric_grid(test_summary, coverage_summary)}
    {render_quality_bars(test_summary, coverage_summary)}
    {render_coverage_table(coverage_summary)}
    {render_test_table(test_summary)}
  </main>
</body>
</html>
"""


def render_metric_grid(
    test_summary: TestSummary,
    coverage_summary: CoverageSummary,
) -> str:
    return f"""
    <div class="grid">
      {metric("Tests", str(test_summary.tests))}
      {metric("Passed", str(test_summary.passed))}
      {metric("Failures", str(test_summary.failures + test_summary.errors))}
      {metric("Line coverage", f"{coverage_summary.line_percent:.2f}%")}
    </div>
"""


def render_quality_bars(
    test_summary: TestSummary,
    coverage_summary: CoverageSummary,
) -> str:
    return f"""
    <section>
      <h2>Quality Summary</h2>
      {progress_row("Pass rate", test_summary.pass_rate, "#238b45")}
      {progress_row("Line coverage", coverage_summary.line_percent, "#2563eb")}
      {progress_row("Branch coverage", coverage_summary.branch_percent, "#7c3aed")}
    </section>
"""


def render_coverage_table(coverage_summary: CoverageSummary) -> str:
    rows = "\n".join(
        f"""
        <tr>
          <td class="path">{escape(file.path)}</td>
          <td class="right">{file.lines_covered}/{file.lines_valid}</td>
          <td class="right">{file.line_rate * 100:.2f}%</td>
          <td>{progress_bar(file.line_rate * 100, coverage_color(file.line_rate * 100))}</td>
        </tr>
        """
        for file in coverage_summary.files
    )
    return f"""
    <section>
      <h2>Coverage By File</h2>
      <table>
        <thead>
          <tr>
            <th>File</th>
            <th class="right">Lines</th>
            <th class="right">Coverage</th>
            <th>Visual</th>
          </tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
    </section>
"""


def render_test_table(test_summary: TestSummary) -> str:
    rows = "\n".join(
        f"""
        <tr>
          <td class="status-text {escape(case.status)}">{escape(case.status)}</td>
          <td class="path">{escape(case.classname)}::{escape(case.name)}</td>
          <td class="right">{case.time:.3f}s</td>
          <td>{escape(shorten(case.message, 140))}</td>
        </tr>
        """
        for case in sorted(test_summary.cases, key=lambda item: (item.status == "passed", item.classname, item.name))
    )
    return f"""
    <section>
      <h2>Test Cases</h2>
      <table>
        <thead>
          <tr>
            <th>Status</th>
            <th>Test</th>
            <th class="right">Time</th>
            <th>Message</th>
          </tr>
        </thead>
        <tbody>{rows}</tbody>
      </table>
    </section>
"""


def metric(label: str, value: str) -> str:
    return f"""
      <div class="metric">
        <div class="label">{escape(label)}</div>
        <div class="value">{escape(value)}</div>
      </div>
"""


def progress_row(label: str, value: float, color: str) -> str:
    return f"""
      <p><strong>{escape(label)}</strong> {value:.2f}%</p>
      {progress_bar(value, color)}
"""


def progress_bar(value: float, color: str) -> str:
    bounded = max(0.0, min(100.0, value))
    return f'<div class="bar" style="--value: {bounded:.2f}%; --color: {escape(color)}"><span></span></div>'


def coverage_color(value: float) -> str:
    if value >= 90:
        return "#238b45"
    if value >= 75:
        return "#b7791f"
    return "#c2410c"


def percentage(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator * 100


def shorten(text: str, max_length: int) -> str:
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def escape(value: object) -> str:
    return html.escape(str(value), quote=True)


if __name__ == "__main__":
    raise SystemExit(main())
