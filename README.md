# RAG Chat Interface

A small Retrieval-Augmented Generation (RAG) project with a chat interface and vector-store backed search. The repo includes a simple Panel-based UI (`app.py`) and utilities in `src/rag`.

**Overview**
- **Purpose**: Combine retrieval from a vector store with generative responses.
- **UI**: Panel application served from `app.py`.

**Prerequisites**
- **Python**: 3.8 or newer
- **Virtual environment**: recommended (venv or conda)

**Installation**
1. Clone the repo and change directory:
	```bash
	git clone <repo-url>
	cd RAG
	```
2. Create and activate a virtual environment:
	```bash
	python3 -m venv .venv
	source .venv/bin/activate
	```
3. Install the project dependencies:
	```bash
	pip install -r requirements.txt
	```
4. (Optional) Install `uvicorn` for running ASGI apps:
	```bash
	pip install "uvicorn[standard]"
	```

**Run the app**
- Panel (recommended for the provided UI):
  ```bash
  panel serve app.py --dev
  ```
  You can also serve multiple notebooks or apps, e.g. `panel serve app.py app2.ipynb --dev`.
- ASGI / Uvicorn (only if your project exposes an ASGI app):
  ```bash
  uvicorn app:app --reload --port 8000
  ```
  Replace `app:app` with `module:app` if your ASGI application object has a different name.

**Notes & Troubleshooting**
- If `panel` command is not found, ensure `panel` is installed and your virtual environment is activated: `pip install panel`.
- For GPU or large-model environments, ensure additional dependencies (transformers, torch, etc.) are installed as needed.
- The vector store files live under `vector_store/` — keep that directory when moving the project.

If you want, I can add example `uvicorn`/systemd service files or a short development Makefile.

