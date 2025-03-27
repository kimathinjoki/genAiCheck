# AI Content Detection Tool

A robust tool for detecting AI-generated content in student submissions by comparing them against multiple LLM responses and analyzing linguistic patterns.

## Overview

This tool analyzes student submissions (DOCX and PDF files) to determine the likelihood that they were generated by AI models. It uses a multi-faceted approach to analyze text patterns, complexity, and similarity to known AI responses.

## Features

- Compares student answers against 7 different LLM models: OpenAI (4o, o3, o1), Anthropic Claude (3.5 sonnet, 3.7 sonnet), Google Gemini (1.5 pro, 2.0-flash)
- Analyzes text patterns, complexity, and structure
- Handles Word DOCX and PDF files
- Generates detailed Excel reports with color-coded results
- Caches API responses to reduce costs and processing time

## Installation

```bash

    # Clone the repository
    git clone https://github.com/yourusername/ai-detection-tool.git
    cd ai-detection-tool

    # Create a virtual environment
    python -m venv venv

    # Activate the virtual environment
    # On Windows:
    venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate

    # Install dependencies
    pip install -r requirements.txt

```

## Configuration

Create a `.env` file with your API keys:

```
    OPENAI_API_KEY=your_openai_key_here
    ANTHROPIC_API_KEY=your_anthropic_key_here
    GOOGLE_API_KEY=your_google_key_here

```

## Usage

1. Place your questions in a text file (one per line)
2. Put student submissions (DOCX/PDF) in a directory
3. Run the tool:

```bash
    python ai_detection.py --questions_doc ./assignment/assignment.docx --submissions ./student_submissions
```

or

```bash
    python ai_detection.py --questions questions.txt --submissions ./resources/student_submissions --output results.xlsx
```

## How It Works

The AI detection tool uses a sophisticated multi-layered approach to identify AI-generated content:

### 1. LLM Response Collection

Each question is sent to seven different large language models (OpenAI, Claude, Gemini) to obtain reference answers. These represent how AI systems typically respond to the question.

### 2. Multi-faceted Analysis

The detection combines several analysis methods:

#### A. Semantic Similarity (30% of the score)
- Uses TF-IDF vectorization to convert texts into numerical representations
- Calculates cosine similarity between student answers and AI-generated responses
- Higher similarity indicates potential AI generation

#### B. Writing Pattern Analysis (30% of the score)
- Detects patterns common in AI writing:
  - Repetitive phrases
  - Formulaic structure (standard intros/conclusions)
  - Conventional transitions
  - Generic examples
  - Overuse of hedging language

#### C. Linguistic Feature Comparison (40% of the score)
- Complexity scores (sentence structure, vocabulary)
- Sentence length patterns
- Word diversity ratios
- Unique words usage

### 3. Weighting System

Each student answer gets compared to all seven AI model outputs, with a weighted scoring system:
- 70% based on direct comparison with AI responses
- 30% based on general AI writing pattern detection

### 4. Final Scoring

The final AI score (0.0-1.0) represents the probability that the answer was AI-generated:
- 0.0-0.5: Likely human-written
- 0.5-0.7: Possibly AI-generated
- 0.7-1.0: Likely AI-generated

The "most_likely_source" field shows which specific AI model the writing most closely resembles, while the "confidence" rating indicates how certain the system is about its assessment based on the consistency and strength of the detected patterns.

## Understanding Results

The tool produces CSV and Excel files with the following columns:

- **question_id**: ID number of the question
- **question**: Text of the question being analyzed
- **student_id**: Student identifier extracted from filename
- **ai_score**: Score from 0.0 to 1.0 indicating likelihood of AI generation
- **most_likely_source**: Which AI model it most resembles (or "Human-written")
- **confidence**: How confident the tool is in its assessment (High/Medium/Low)

## Command Line Options

- `--questions`: Path to file containing questions (required)
- `--questions_doc`: Path to DOCX or PDF file containing questions (alternative to --questions)
- `--submissions`: Directory containing student files (required)
- `--output`: Output Excel file (default: ai_detection_results.xlsx)
- `--no-cache`: Disable caching of LLM responses
- `--verbose`: Print detailed progress information

## Limitations

- The tool provides probabilities, not definitive answers
- False positives are possible, especially with highly technical content
- Students who regularly use AI tools for writing practice may adopt AI-like writing patterns
- Results should be used as part of a broader assessment strategy, not as the sole determinant

## License

[MIT License](LICENSE)