from pathlib import Path


def load_prompt_template(name: str) -> str:
    """Load prompt template from prompts directory."""
    prompts_dir = Path(__file__).parent
    template_path = prompts_dir / f"{name}.txt"
    return template_path.read_text()
