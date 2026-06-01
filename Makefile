.PHONY: help test install clean ask bench

help:		## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

test:		## Run the offline test suite (no API key needed)
	cd .. && python -m unittest discover -s self_rag/tests -t . -v

install:	## Editable install of the package
	pip install -e .

ask:		## Answer a question: make ask ARGS='"Which planet is the largest?" --show-trace'
	python -m self_rag $(ARGS)

bench:		## Evaluate answer accuracy + abstention on the bundled eval set
	python -m self_rag --bench

clean:		## Remove caches and build artifacts
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf *.egg-info build dist .eggs
