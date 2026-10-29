import platform
import secrets

from src.prng.base import BaseGenerator, Sample


class CSPRNGGenerator(BaseGenerator):
    name = "csprng"
    label = 0

    def generate_sample(self, n_bytes: int) -> Sample:
        return Sample(data=secrets.token_bytes(n_bytes), seed=None)

    def params(self) -> dict:
        return {
            "source": "secrets.token_bytes",
            "os": platform.system(),
            "os_release": platform.release(),
            "python": platform.python_version(),
        }
