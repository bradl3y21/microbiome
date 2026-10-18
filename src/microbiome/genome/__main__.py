"""Print the two-genome phenotype demonstration as deterministic JSON."""

import json

from .phenotype import phenotype_demo


def main() -> None:
    print(json.dumps(phenotype_demo(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
