from pathlib import Path

def get_resolved_path(given_path: Path) -> str:
    # Get path to current file's directory
    current_dir = Path(__file__).parent
    # Resolve relative path from current file
    model_path = (current_dir / given_path).resolve()
    return model_path

def get_text_prompts():
    """A method to retrieve text prompts for GLIP

    Returns:
        _type_: _description_
    """
    TEXT_PROMPTS = [
        "a transparent cylindrical 8 oz drinking glass",
        "a transparent cylindrical 12 oz drinking glass",
        "a transparent conical 12 oz glass",
        "a plate",
        "a transparent 16 oz cylindrical glass",
        "a ceramic coffee mug with a handle",
        "a stainless steel saucepan",
        "a pressure cooker with black handles",
        "a kitchen sink",
    ]
    return TEXT_PROMPTS