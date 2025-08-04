# My Python Project
A Python script for NLP and web scraping tasks.

## Setup
1. Clone: `git clone https://github.com/mirriam/my-python-project.git`
2. Create virtual environment: `python3 -m venv venv`
3. Activate: `source venv/bin/activate`
4. Install: `pip install -r requirements.txt`
5. Install NLTK data: `python -c "import nltk; nltk.download('punkt')"`
6. Install Java: `sudo apt-get install openjdk-11-jre`
7. Set token: `export HF_TOKEN=your-hf-token` or use `.env`
8. Run: `python scripts/script.py`

## Automation
- GitHub Actions runs on push to `main`.
- Set `HF_TOKEN` in Actions secrets.
