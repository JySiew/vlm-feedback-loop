class Skills:
    _skills: list
    
    def __init__(self, skills: list):
        self._skills = skills

    def get_skills(self):
        return self._skills
    
    def set_skills(self, skills: list):
        self._skills = skills
    
    def load_skills_from_file(self, file_path: str):
        """Load skills from a text file, one skill per line."""
        try:
            with open(file_path, 'r') as file:
                self._skills = [line.strip() for line in file if line.strip()]
        except FileNotFoundError:
            raise FileNotFoundError(f"Skills file not found: {file_path}")