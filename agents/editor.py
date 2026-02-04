import json
import os
from typing import Dict, List

from src.agents.base_agent import BaseAgent
from src.utils.llm_client import LLMClient


EDITOR_SYSTEM_PROMPT = """
You are an expert Slidev Developer and Presentation Designer.
Transform raw text into beautiful, structured Slidev markdown slides.

Hard constraints you MUST follow:
1) NO duplicate slides. If two slides have near-identical titles or bullets, merge them.
2) Cover slide: include an explicit cover slide in the body using Slidev frontmatter:
   ---
   layout: cover
   ---
   Then use the following template (do not put this only in global frontmatter):
   - Title: ...
   - Author: ...
   - Date: ...
   - Source: ...
3) Keep text density low: prefer 3-5 bullets per slide, <= 2 lines per bullet.
4) Each slide must have a clear title and distinct purpose.
5) Prefer visual hierarchy: clear title, short bullets, emphasize keywords with **bold**.
6) Avoid tiny fonts: if too much text, split into multiple slides.

When refining, you MUST respond with a JSON object:
{
  "slides_md": "<full Slidev markdown>",
  "adjustments": [
    {
      "page_index": 3,
      "issue": "Content density",
      "change": "Split into two slides and reduced bullets to 4"
    }
  ],
  "summary": {
    "quality": "functional | solid | professional",
    "remaining_risks": ["..."],
    "notes": "..."
  }
}
Follow Slidev syntax strictly in slides_md and output valid JSON only.
""".strip()


class EditorAgent(BaseAgent):
    def __init__(self, model_name: str = "gpt-5.1", provider: str | None = None):
        provider = provider or os.getenv("EDITOR_LLM_PROVIDER") or "deepseek"
        super().__init__(role="Editor", model_name=model_name, provider=provider)
        self.set_system_prompt(EDITOR_SYSTEM_PROMPT)
    
    def self_review(self, slides_md: str, image_paths: List[str] | None = None) -> Dict:
        """Self-review slides using the same model."""
        prompt = """
        Review your own Slidev markdown slides for:
        1. Duplicate slides
        2. Text density issues
        3. Layout problems
        4. Typography or readability issues
        5. Inconsistent formatting
        6. Missing cover metadata

        Return feedback in JSON format using the same structure as CriticAgent with a summary section.
        """
        if image_paths and self.llm_client.supports_vision():
            response = self.chat(prompt, image_paths=image_paths, json_mode=True)
        else:
            response = self.chat(f"{prompt}\n\nSlides:\n{slides_md}", json_mode=True)
        payload = LLMClient.safe_json_loads(response.content)
        if isinstance(payload, dict) and "feedback" in payload:
            payload.setdefault("summary", {})
            return payload
        if isinstance(payload, list):
            return {"feedback": payload, "summary": {}}
        try:
            parsed = json.loads(response.content)
            if isinstance(parsed, dict) and "feedback" in parsed:
                parsed.setdefault("summary", {})
                return parsed
            if isinstance(parsed, list):
                return {"feedback": parsed, "summary": {}}
        except json.JSONDecodeError:
            return {"feedback": [], "summary": {}}
        return {"feedback": [], "summary": {}}

    def generate_draft(self, raw_content: str) -> str:
        prompt = (
            "Please convert the following content into Slidev markdown slides. "
            "Return the full slides.md content only.\n\n"
            "Remember: no duplicate slides; include a cover slide in the body using Slidev frontmatter `---\\nlayout: cover\\n---` and bullets for Title/Author/Date/Source; "
            "avoid dense paragraphs.\n\n"
            f"Content:\n{raw_content}\n"
        )
        response = self.chat(prompt)
        return response.content.strip()

    @staticmethod
    def _fallback_adjustments(feedback: List[Dict]) -> Dict:
        adjustments = []
        for item in feedback:
            adjustments.append(
                {
                    "page_index": item.get("page_index", "N/A"),
                    "issue": item.get("issue", "Unknown"),
                    "change": "Planned adjustment based on feedback",
                }
            )
        return {"slides_md": "", "adjustments": adjustments, "summary": {}}

    def refine_slides(self, current_code: str, feedback: List[Dict]) -> Dict:
        feedback_text = json.dumps(feedback, ensure_ascii=False, indent=2)
        prompt = (
            "Please refine the Slidev markdown according to the feedback. "
            "Return JSON only using the specified structure with slides_md, adjustments, summary.\n\n"
            "Critical rules: eliminate duplicates; include a cover slide in the body using Slidev frontmatter `---\\nlayout: cover\\n---` with bullets for Title/Author/Date/Source; "
            "reduce text density and keep each slide distinct.\n\n"
            f"Current Slides:\n{current_code}\n\n"
            f"Feedback (JSON):\n{feedback_text}\n"
        )
        response = self.chat(prompt, json_mode=True)
        payload = LLMClient.safe_json_loads(response.content)
        if isinstance(payload, dict) and "slides_md" in payload:
            payload.setdefault("adjustments", [])
            payload.setdefault("summary", {})
            return payload
        if payload is None:
            try:
                parsed = json.loads(response.content)
                if isinstance(parsed, dict) and "slides_md" in parsed:
                    parsed.setdefault("adjustments", [])
                    parsed.setdefault("summary", {})
                    return parsed
            except json.JSONDecodeError:
                pass
        fallback = self._fallback_adjustments(feedback)
        fallback["slides_md"] = response.content.strip()
        return fallback
