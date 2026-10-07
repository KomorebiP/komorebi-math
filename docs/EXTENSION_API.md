# Extension API

Future packages can register themselves through the entry point group:

```toml
[project.entry-points."komorebi_math.extensions"]
my_extension = "komorebi_my_extension:extension"
```

The entry point must provide an `ExtensionSpec` or a zero argument callable that returns one.

```python
from komorebi_math.extensions import ExtensionSpec


def register() -> None:
    pass


extension = ExtensionSpec(
    name="komorebi-example",
    version="0.1.0",
    api_version="1",
    modules=("komorebi_example",),
    optional_dependencies=(),
    description="Example extension",
    register=register,
)
```

The command:

```powershell
komorebi doctor
```

loads each extension in isolation and reports failures without preventing the core package from importing.

Extensions must not silently change the meaning of a core object. New backends and feature records must be registered under distinct names. Conditional mathematical constructions must return conditional certificates with their unmet assumptions.
