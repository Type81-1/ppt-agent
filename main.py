import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import List

import typer
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils.llm_client import LLMClient

from src.agents.editor import EditorAgent
from src.agents.critic import CriticAgent
from src.utils.slidev_runner import SlidevRunner, RenderError


app = typer.Typer(add_completion=False)


def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_text_file(path: str, content: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def clear_dir(path: str) -> None:
    if Path(path).exists():
        shutil.rmtree(path)
    Path(path).mkdir(parents=True, exist_ok=True)


def strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip() + "\n"
    return text


def fix_frontmatter_aliases(text: str) -> str:
    if not text.startswith("---"):
        return text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return text
    _, frontmatter, body = parts[0], parts[1], parts[2]
    fixed_lines = []
    for line in frontmatter.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            fixed_lines.append(line)
            continue
        if ":" not in line:
            fixed_lines.append(line)
            continue
        key, value = line.split(":", 1)
        value_stripped = value.strip()
        if value_stripped.startswith("*") and not (
            value_stripped.startswith("'*") or value_stripped.startswith('"*')
        ):
            fixed_lines.append(f"{key}: \"{value_stripped}\"")
        else:
            fixed_lines.append(line)
    fixed_frontmatter = "\n".join(fixed_lines)
    return f"---\n{fixed_frontmatter}\n---{body}"


def fix_cover_frontmatter(text: str) -> str:
    """Ensure cover slide uses proper Slidev frontmatter delimiters."""
    if "layout: cover" not in text:
        return text
    lines = text.splitlines()
    fixed_lines: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped == "layout: cover":
            prev_line = lines[i - 1].strip() if i > 0 else ""
            if prev_line != "---":
                fixed_lines.append("---")
            fixed_lines.append("layout: cover")
            i += 1
            # Skip immediate blank lines that can break frontmatter parsing
            while i < len(lines) and lines[i].strip() == "":
                i += 1
            next_line = lines[i].strip() if i < len(lines) else ""
            if next_line != "---":
                fixed_lines.append("---")
            continue
        fixed_lines.append(line)
        i += 1
    return "\n".join(fixed_lines)


def extract_slides_md(content: str) -> str:
    """Extract markdown from JSON-wrapped responses when present."""
    if not content:
        return content
    raw = content.strip()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and isinstance(parsed.get("slides_md"), str):
            return parsed["slides_md"]
    except json.JSONDecodeError:
        pass
    return content

def is_approved(feedback: List[dict]) -> bool:
    return not feedback


def extract_feedback(payload: dict | List[dict]) -> List[dict]:
    if isinstance(payload, dict):
        return payload.get("feedback", [])
    if isinstance(payload, list):
        return payload
    return []


def extract_summary(payload: dict | List[dict]) -> dict:
    if isinstance(payload, dict):
        return payload.get("summary", {})
    return {}


def parse_report_metrics(report_path: str) -> dict:
    if not os.path.exists(report_path):
        return {}
    metrics = {}
    with open(report_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("Iterations:"):
                metrics["iterations"] = stripped.split(":", 1)[1].strip()
            elif stripped.startswith("Elapsed Time:"):
                metrics["elapsed"] = stripped.split(":", 1)[1].strip()
            elif stripped.startswith("- Final Slide Count:"):
                metrics["slides"] = stripped.split(":", 1)[1].strip()
            elif stripped.startswith("- Total Input Tokens:"):
                metrics["input_tokens"] = stripped.split(":", 1)[1].strip()
            elif stripped.startswith("- Total Output Tokens:"):
                metrics["output_tokens"] = stripped.split(":", 1)[1].strip()
            elif stripped.startswith("- Estimated Cost:"):
                metrics["cost"] = stripped.split(":", 1)[1].strip()
    return metrics


@app.command()
def run(
    input_path: str = "data/paper_summary.txt",
    output_dir: str = "output",
    max_iterations: int = 5,
    model_name: str = "gpt-5.1",
    mode: str = typer.Option("dual", help="Mode: 'dual' (Editor+Critic) or 'single' (Editor self-review)"),
):
    """Run the PPT-Agent pipeline."""
    load_dotenv(dotenv_path=ROOT_DIR / ".env")
    chromium_env = os.getenv("PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH")
    if chromium_env:
        chromium_env = chromium_env.strip().strip('"')
        os.environ["PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH"] = chromium_env
    else:
        custom_chrome_path = os.getenv("CHROME_CUSTOM_PATH")
        if custom_chrome_path:
            os.environ["PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH"] = custom_chrome_path.strip().strip('"')
    start_time = time.time()

    input_path_resolved = Path(input_path)
    if not input_path_resolved.is_absolute():
        input_path_resolved = ROOT_DIR / input_path_resolved
    output_dir_resolved = Path(output_dir)
    if not output_dir_resolved.is_absolute():
        output_dir_resolved = ROOT_DIR / output_dir_resolved

    raw_content = read_text_file(str(input_path_resolved))

    default_model = model_name or "gpt-5.1"

    editor_provider = os.getenv("EDITOR_LLM_PROVIDER") or os.getenv("LLM_PROVIDER", "openai")
    critic_provider = os.getenv("CRITIC_LLM_PROVIDER") or os.getenv("LLM_PROVIDER", "openai")

    editor_model = os.getenv("EDITOR_LLM_MODEL") or os.getenv("LLM_MODEL") or default_model
    critic_model = os.getenv("CRITIC_LLM_MODEL") or os.getenv("LLM_MODEL") or default_model

    if os.getenv("EDITOR_LLM_MODEL") is None and os.getenv("LLM_MODEL") is None:
        if editor_provider == "deepseek":
            editor_model = "deepseek-chat"
        elif editor_provider == "moonshot":
            editor_model = "moonshot-v1-8k"

    if os.getenv("CRITIC_LLM_MODEL") is None and os.getenv("LLM_MODEL") is None:
        if critic_provider == "deepseek":
            critic_model = "deepseek-chat"
        elif critic_provider == "moonshot":
            critic_model = "moonshot-v1-8k"

    editor = EditorAgent(model_name=editor_model, provider=editor_provider)
    critic = None
    if mode == "dual":
        critic = CriticAgent(model_name=critic_model, provider=critic_provider)
    runner = SlidevRunner(work_dir=str(Path(__file__).resolve().parents[1]))

    if mode == "dual":
        base_output_dir = ROOT_DIR / "multi_output"
        current_dir = os.path.join(str(base_output_dir), "current")
        history_dir = os.path.join(str(base_output_dir), "history")
        logs_dir = os.path.join(str(base_output_dir), "logs")
        images_dir = os.path.join(current_dir, "images")
    else:
        base_output_dir = ROOT_DIR / "single_output"
        current_dir = str(base_output_dir)
        history_dir = os.path.join(str(base_output_dir), "history")
        logs_dir = os.path.join(str(base_output_dir), "logs")
        images_dir = os.path.join(current_dir, "images")
    ensure_dir(current_dir)
    ensure_dir(history_dir)
    ensure_dir(logs_dir)

    feedback_payload: dict | List[dict] = []
    feedback: List[dict] = []
    critic_summary: dict = {}
    slides_md = ""
    last_rendered_images: List[str] = []
    last_render_error: str | None = None

    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    run_log_path = os.path.join(logs_dir, f"run_{run_stamp}.log")
    
    # Track usage metrics
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0.0

    def append_run_log(message: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {message}\n"
        with open(run_log_path, "a", encoding="utf-8") as f:
            f.write(line)
        typer.echo(message)

    chromium_log_path = os.getenv("PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH")
    if chromium_log_path:
        append_run_log(f"Using Chromium from env: {chromium_log_path}")
    else:
        append_run_log("No PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH found in .env")
    
    # Install Slidev CLI if not already installed
    append_run_log("Installing Slidev CLI and Playwright (may take a few minutes)...")
    try:
        append_run_log("Installing npm dependencies...")
        runner.install_dependencies()
        append_run_log("Slidev CLI installed successfully")
    except Exception as e:
        append_run_log(f"Slidev installation failed: {str(e)}. Will try to use npx fallback")

    for iteration in range(1, max_iterations + 1):
        append_run_log(f"Iteration {iteration}/{max_iterations} started")
        editor_adjustments: List[dict] = []
        editor_summary: dict = {}
        if iteration == 1:
            append_run_log("Editor: generating draft")
            slides_md = editor.generate_draft(raw_content)
        else:
            append_run_log("Editor: refining slides")
            refinement = editor.refine_slides(slides_md, feedback)
            slides_md = refinement.get("slides_md", "") if isinstance(refinement, dict) else str(refinement)
            editor_adjustments = refinement.get("adjustments", []) if isinstance(refinement, dict) else []
            editor_summary = refinement.get("summary", {}) if isinstance(refinement, dict) else {}

        slides_md = extract_slides_md(slides_md)
        slides_md = strip_code_fence(slides_md)
        slides_md = fix_frontmatter_aliases(slides_md)
        slides_md = fix_cover_frontmatter(slides_md)

        editor_log_path = os.path.join(logs_dir, f"iter_{iteration}_editor.txt")
        editor_output = {
            "slides_md": slides_md,
            "adjustments": editor_adjustments,
            "summary": editor_summary,
            "raw_response": editor.last_response,
        }
        write_text_file(editor_log_path, json.dumps(editor_output, ensure_ascii=False, indent=2))
        append_run_log(f"Editor output saved to {editor_log_path}")
        
        # Track usage
        if hasattr(editor, 'last_response_usage') and editor.last_response_usage:
            total_input_tokens += editor.last_response_usage.get('prompt_tokens', 0)
            total_output_tokens += editor.last_response_usage.get('completion_tokens', 0)
            total_cost += LLMClient.calculate_context_cost(editor.last_response_usage)

        slides_path = os.path.join(current_dir, "slides.md")
        write_text_file(slides_path, slides_md)
        rendered_log_path = os.path.join(logs_dir, f"iter_{iteration}_rendered.md")
        write_text_file(rendered_log_path, slides_md)
        append_run_log(f"Rendered markdown saved to {rendered_log_path}")

        append_run_log("Rendering slides to images (Slidev export)...")
        clear_dir(images_dir)
        image_paths = []
        render_error = None
        try:
            image_paths = runner.render_slides(slides_path, images_dir)
        except RenderError as e:
            render_error = str(e)
            append_run_log("Render failed. Sending error back to editor for fixes")
        last_rendered_images = image_paths
        last_render_error = render_error

        if render_error:
            feedback = [
                {
                    "issue": "Render Error",
                    "details": render_error,
                    "severity": "CRITICAL",
                }
            ]
            critic_log_path = os.path.join(logs_dir, f"iter_{iteration}_critic.txt")
            write_text_file(
                critic_log_path,
                json.dumps({"feedback": feedback, "summary": {}}, ensure_ascii=False, indent=2),
            )
            append_run_log(f"Render error logged to {critic_log_path}")
        else:
            if mode == "dual":
                append_run_log("Critic: reviewing slides")
                feedback_payload = critic.review(image_paths, slides_md=slides_md) if critic else {}
                feedback = extract_feedback(feedback_payload)
                critic_summary = extract_summary(feedback_payload)
                critic_log_path = os.path.join(logs_dir, f"iter_{iteration}_critic.txt")
                critic_output = {
                    "feedback": feedback,
                    "summary": critic_summary,
                    "raw_response": critic.last_response if critic else None,
                }
                write_text_file(critic_log_path, json.dumps(critic_output, ensure_ascii=False, indent=2))
                append_run_log(f"Critic output saved to {critic_log_path}")
            else:
                append_run_log("Editor: self-reviewing slides")
                feedback_payload = editor.self_review(slides_md, image_paths)
                feedback = extract_feedback(feedback_payload)
                critic_summary = extract_summary(feedback_payload)
                critic_log_path = os.path.join(logs_dir, f"iter_{iteration}_critic.txt")
                write_text_file(
                    critic_log_path,
                    json.dumps(
                        {"feedback": feedback, "summary": critic_summary},
                        ensure_ascii=False,
                        indent=2,
                    ),
                )
                append_run_log(f"Self-review output saved to {critic_log_path}")
            
            # Display feedback in terminal
            if feedback:
                append_run_log("\n📋 Critic Feedback:")
                for idx, issue in enumerate(feedback, 1):
                    severity = issue.get("severity", "UNKNOWN")
                    page = issue.get("page_index", "N/A")
                    issue_desc = issue.get("issue", "No description")
                    suggestion = issue.get("suggestion", "No suggestion")
                    append_run_log(f"{idx}. [{severity}] Page {page}: {issue_desc}")
                    append_run_log(f"   Suggestion: {suggestion}")
                append_run_log("\n")

        iter_dir = os.path.join(history_dir, f"iter_{iteration}")
        ensure_dir(iter_dir)
        shutil.copy(slides_path, os.path.join(iter_dir, "slides.md"))
        if image_paths:
            images_history = os.path.join(iter_dir, "images")
            clear_dir(images_history)
            for img in image_paths:
                shutil.copy(img, images_history)

        critique_path = os.path.join(iter_dir, "critique.json")
        with open(critique_path, "w", encoding="utf-8") as f:
            json.dump(
                {"feedback": feedback, "summary": critic_summary},
                f,
                ensure_ascii=False,
                indent=2,
            )

        if is_approved(feedback):
            append_run_log("Approved. Stopping iterations.")
            break
        append_run_log("Not approved. Continuing to next iteration.")

    elapsed = time.time() - start_time
    typer.echo(f"Done. Final slides at {current_dir}/slides.md")
    if last_render_error:
        typer.echo("❌ Render failed. Please check output/logs/iter_*_critic.txt")
    else:
        typer.echo(f"✅ Rendered {len(last_rendered_images)} slide images in {images_dir}")
    typer.echo(f"Elapsed: {elapsed:.2f}s")
    # Print usage metrics
    typer.echo(f"- Total Input Tokens: {total_input_tokens}")
    typer.echo(f"- Total Output Tokens: {total_output_tokens}")
    typer.echo(f"- Estimated Cost: ${total_cost:.4f}")
    # Write metrics to run log
    with open(run_log_path, "a", encoding="utf-8") as f:
        f.write("\n=== Usage Metrics ===\n")
        f.write(f"- Total Input Tokens: {total_input_tokens}\n")
        f.write(f"- Total Output Tokens: {total_output_tokens}\n")
        f.write(f"- Estimated Cost: ${total_cost:.4f}\n")
    
    # Generate comparison report (single vs dual)
    report_path = os.path.join(logs_dir, f"{mode}_mode_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"PPT-Agent Report - {mode.capitalize()} Mode\n")
        f.write(f"Run Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Iterations: {iteration}\n")
        f.write(f"Elapsed Time: {elapsed:.2f}s\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Input File: {input_path}\n")
        f.write(f"Output Directory: {output_dir}\n")
        f.write("\n---\n")
        f.write("Key Metrics:\n")
        f.write(f"- Total Iterations: {iteration}\n")
        f.write(f"- Final Slide Count: {len(last_rendered_images)}\n")
        f.write(f"- Mode: {mode}\n")
        f.write(f"- Total Input Tokens: {total_input_tokens}\n")
        f.write(f"- Total Output Tokens: {total_output_tokens}\n")
        f.write(f"- Estimated Cost: ${total_cost:.4f}\n")
        if critic_summary:
            f.write("\n---\n")
            f.write("Final Critic Summary:\n")
            f.write(json.dumps(critic_summary, ensure_ascii=False, indent=2))
            f.write("\n")
        other_mode = "single" if mode == "dual" else "dual"
        other_report_path = os.path.join(logs_dir, f"{other_mode}_mode_report.txt")
        other_metrics = parse_report_metrics(other_report_path)
        if other_metrics:
            f.write("\n---\n")
            f.write(f"Comparison vs {other_mode} mode:\n")
            f.write(f"- {mode} iterations: {iteration} vs {other_metrics.get('iterations', 'N/A')}\n")
            f.write(f"- {mode} elapsed: {elapsed:.2f}s vs {other_metrics.get('elapsed', 'N/A')}\n")
            f.write(f"- {mode} slides: {len(last_rendered_images)} vs {other_metrics.get('slides', 'N/A')}\n")
            f.write(f"- {mode} input tokens: {total_input_tokens} vs {other_metrics.get('input_tokens', 'N/A')}\n")
            f.write(f"- {mode} output tokens: {total_output_tokens} vs {other_metrics.get('output_tokens', 'N/A')}\n")
            f.write(f"- {mode} cost: ${total_cost:.4f} vs {other_metrics.get('cost', 'N/A')}\n")
    typer.echo(f"📊 Report generated at {report_path}")


if __name__ == "__main__":
    app()
