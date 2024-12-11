help:  ## Show help
	@grep -E '^[.a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

clean: ## Clean autogenerated files
	rm -rf dist
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	rm -f .coverage

docker-build: ## Build docker images
	docker compose build --no-cache

docker-up: ## Start docker containers
	docker compose up -d

docker-down: ## Stop docker containers
	docker compose down

docker-attach: ## Attach to development container
	docker compose exec dev bash

format: ## Run pre-commit hooks
	poetry run pre-commit run -a

test: ## Run tests
	poetry run pytest -v

type: ## Run type check
	poetry run pyright
