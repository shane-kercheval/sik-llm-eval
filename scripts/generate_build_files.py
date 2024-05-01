from pydantic import BaseModel
import tomlkit
from ruamel.yaml import YAML
from pathlib import Path
from typing import List, Dict, Any, NoReturn


class TemplateConfig(BaseModel):
    """Configuration used to instantiate TemplateManager subclasses."""

    template_path: Path
    output_path: Path
    warning_header: str
    project_name: str
    project_version: str


class TemplateManager:
    """
    Abstract base class for managing template files and updating project configurations.
    Handles generic template operations and sets a framework for derived classes.
    """

    def __init__(self, config: TemplateConfig):
        """Initialize the TemplateManager with paths for the template and the output."""
        self.template_path = config.template_path
        self.output_path = config.output_path
        self.warning_header = config.warning_header
        self.project_name = config.project_name
        self.project_version = config.project_version

    def load_template(self) -> Dict[str, Any]:
        """Load the template data from a file. This method should be implemented by subclasses."""
        raise NotImplementedError

    def save_updated_file(self, content: Dict[str, Any]) -> NoReturn:
        """Save the updated content to a file. This method should be implemented by subclasses."""
        raise NotImplementedError

    def update_dependencies(self, dependencies: List[str]) -> NoReturn:
        """
        Update the dependency list in the template.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError


class PyProjectManager(TemplateManager):
    """Manages 'pyproject.toml' files for Python projects, inheriting from TemplateManager."""

    def load_template(self) -> Dict[str, Any]:
        """Load a TOML formatted template file and return the content as a dictionary."""
        with open(self.template_path, encoding="utf-8") as file:
            return tomlkit.parse(file.read())

    def save_updated_file(self, content: Dict[str, Any]) -> None:
        """Save the updated TOML content to the output file."""
        with open(self.output_path, "w", encoding="utf-8") as file:
            tomlkit.dump(tomlkit.comment(self.warning_header), file)
            tomlkit.dump(content, file)

    def update_dependencies(self, dependencies: List[str]) -> None:
        """Update the 'pyproject.toml' file with the specified dependencies."""
        content = self.load_template()

        content["project"]["name"] = self.project_name
        content["project"]["version"] = self.project_version

        updated_deps = sorted(set(content["project"]["dependencies"]).union(dependencies))
        content["project"]["dependencies"] = list(updated_deps)

        self.save_updated_file(content)


class YamlManager(TemplateManager):
    """Manages YAML files for project configurations, inheriting from TemplateManager."""

    def __init__(self, config: TemplateConfig):
        """Initialize the YamlManager with YAML specific settings along with the paths."""
        super().__init__(config)
        self.yaml = YAML()
        self.yaml.preserve_quotes = True
        self.yaml.indent(mapping=2, sequence=4, offset=2)

    def load_template(self) -> Dict[str, Any]:
        """Load the YAML formatted template file and return the content as a dictionary."""
        with open(self.template_path) as file:
            return self.yaml.load(file)

    def save_updated_file(self, content: Dict[str, Any]) -> None:
        """Save the updated YAML content to the output file."""
        with open(self.output_path, "w") as file:
            self.yaml.dump(content, file)

    def update_dependencies(self, dependencies: List[str]) -> None:
        """
        Update the 'environment.yml' file for conda
        with the specified dependencies.
        """
        content = self.load_template()
        content.yaml_set_start_comment(self.warning_header)

        content["name"] = self.project_name
        updated_deps = sorted(set(content["dependencies"]).union(dependencies))
        content["dependencies"] = list(updated_deps)

        self.save_updated_file(content)


class CondaMetaManager(YamlManager):
    """Manages 'meta.yaml' files specifically for building conda packages."""

    def update_dependencies(self, dependencies: List[str]) -> None:
        """
        Update the 'meta.yaml' file for building a conda package
        with the specified dependencies.
        """
        content = self.load_template()
        content.yaml_set_start_comment(self.warning_header)

        content["package"]["name"] = self.project_name
        content["package"]["version"] = self.project_version

        existing_deps = set(content["requirements"]["run"])
        updated_deps = sorted(existing_deps.union(dependencies))
        content["requirements"]["run"] = list(updated_deps)

        self.save_updated_file(content)


def pip_to_conda_version(pip_version: str) -> str:
    """
    Convert a pip-style version string to a conda-compatible version string.
    This conversion replaces '~=' with '=' and '==' with '=', for conda's versioning syntax.
    """
    return pip_version.replace("~=", "=").replace("==", "=").replace("torch", "pytorch")


def load_dependencies(file_path: Path) -> List[str]:
    """
    Load dependencies from a given file path.
    Each line in the file should contain a single dependency string in pip-compatible format.
    """
    with open(file_path) as file:
        return [line.strip() for line in file if line.strip()]


def _build_config(template_path: Path, output_path: Path) -> TemplateConfig:
    """Helper function to build config with preset values."""
    return TemplateConfig(
        template_path=template_path,
        output_path=output_path,
        warning_header="DO NOT EDIT THIS FILE DIRECTLY AS IT IS GENERATED",
        project_name="llm-eval",
        project_version="0.0.1",
    )


def main() -> None:
    """Run main function for generating build files."""
    root_dir = Path(__file__).resolve().parent.parent
    conda_recipe_dir = root_dir / "conda.recipe"
    template_dir = root_dir / "scripts" / "templates"

    dependencies_path = root_dir / "requirements.txt"
    dependencies = load_dependencies(dependencies_path)

    pyproject_manager = PyProjectManager(
        _build_config(
            template_path=template_dir / "pyproject.toml",
            output_path=root_dir / "pyproject.toml",
        ),
    )
    pyproject_manager.update_dependencies(dependencies)

    environment_manager = YamlManager(
        _build_config(
            template_path=template_dir / "environment.yml",
            output_path=root_dir / "environment.yml",
        ),
    )
    environment_manager.update_dependencies([pip_to_conda_version(dep) for dep in dependencies])

    meta_manager = CondaMetaManager(
        _build_config(
            template_path=template_dir / "meta.yaml",
            output_path=conda_recipe_dir / "meta.yaml",
        ),
    )
    meta_manager.update_dependencies([pip_to_conda_version(dep) for dep in dependencies])


if __name__ == "__main__":
    main()
