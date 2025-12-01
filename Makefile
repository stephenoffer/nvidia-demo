.PHONY: help test lint format install clean docs build docker-build docker-push deploy-staging deploy-production validate lint-all health-check backup setup-dev terraform-init helm-install helm-upgrade helm-uninstall

help:
	@echo "Multimodal Data Pipeline - Available targets:"
	@echo ""
	@echo "Development:"
	@echo "  install         - Install package and dependencies"
	@echo "  setup-dev       - Setup development environment"
	@echo "  test            - Run unit tests"
	@echo "  test-cov        - Run tests with coverage"
	@echo "  test-integration - Run integration tests"
	@echo "  lint            - Run linters"
	@echo "  lint-all        - Run all linting checks"
	@echo "  format          - Format code"
	@echo "  validate        - Validate configuration"
	@echo "  health-check    - Run project health checks"
	@echo "  docs            - Generate documentation"
	@echo "  clean           - Clean build artifacts"
	@echo ""
	@echo "Build:"
	@echo "  build           - Build Python package"
	@echo "  docker-build    - Build Docker image"
	@echo "  docker-push     - Push Docker image to registry"
	@echo ""
	@echo "Deployment:"
	@echo "  deploy-staging  - Deploy to staging environment"
	@echo "  deploy-prod     - Deploy to production environment"
	@echo "  smoke-tests     - Run smoke tests"
	@echo "  backup          - Backup configurations"
	@echo ""
	@echo "Infrastructure:"
	@echo "  terraform-init  - Initialize Terraform"
	@echo "  helm-install    - Install Helm chart"
	@echo "  helm-upgrade    - Upgrade Helm chart"
	@echo "  helm-uninstall  - Uninstall Helm chart"

install:
	pip install -e ".[dev]"

test:
	pytest tests/unit/ -v

test-cov:
	pytest tests/unit/ -v --cov=pipeline --cov-report=html --cov-report=term

test-integration:
	pytest tests/integration/ -v

test-benchmarks:
	pytest tests/benchmarks/ --benchmark-only

lint:
	ruff check pipeline/
	mypy pipeline/ --ignore-missing-imports

format:
	ruff format pipeline/

docs:
	@echo "Documentation is in docs/ directory"

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -r {} +
	find . -type f -name "*.pyc" -delete

build:
	python setup.py sdist bdist_wheel

docker-build:
	docker build -t multimodal-pipeline:latest .

docker-push:
	@if [ -z "$(REGISTRY_URL)" ]; then \
		echo "Error: REGISTRY_URL environment variable not set"; \
		exit 1; \
	fi
	docker tag multimodal-pipeline:latest $(REGISTRY_URL)/multimodal-pipeline:latest
	docker push $(REGISTRY_URL)/multimodal-pipeline:latest

deploy-staging:
	./scripts/deploy.sh staging latest

deploy-prod:
	@echo "WARNING: Deploying to production!"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		./scripts/deploy.sh production latest; \
	else \
		echo "Deployment cancelled"; \
	fi

smoke-tests:
	./scripts/smoke-tests.sh staging

validate:
	./scripts/validate-config.sh

lint-all:
	./scripts/lint-all.sh

health-check:
	./scripts/health-check.sh

backup:
	./scripts/backup.sh

setup-dev:
	./scripts/setup-dev.sh

terraform-init:
	./scripts/terraform-init.sh staging

helm-install:
	helm install multimodal-pipeline deployment/helm/ -n pipeline-production --create-namespace

helm-upgrade:
	helm upgrade multimodal-pipeline deployment/helm/ -n pipeline-production

helm-uninstall:
	helm uninstall multimodal-pipeline -n pipeline-production

deploy-aws:
	./scripts/deploy-aws.sh production us-east-1

deploy-on-prem:
	./scripts/deploy-on-prem.sh production

setup-irsa:
	./scripts/setup-irsa.sh

validate-k8s:
	./scripts/validate-k8s-config.sh

backup-k8s:
	./scripts/backup-k8s.sh pipeline-production

restore-k8s:
	@echo "Usage: make restore-k8s BACKUP_FILE=backup.tar.gz"
	@if [ -z "$(BACKUP_FILE)" ]; then \
		echo "Error: BACKUP_FILE not set"; \
		exit 1; \
	fi
	./scripts/restore-k8s.sh $(BACKUP_FILE) pipeline-production

install-operators:
	./deployment/operators/install-operators.sh

install-gitops:
	kubectl apply -f deployment/gitops/argocd-application.yaml || \
	kubectl apply -f deployment/gitops/flux-app.yaml

validate-operators:
	kubectl get pods -n operators
	kubectl get pods -n cert-manager
	kubectl get pods -n monitoring
	kubectl get pods -n observability

