# SkillHighlight Analyzer User Manual

This guide is for people who will use the SkillHighlight Analyzer website to review a resume. It explains how to open the app, submit a resume, read the results, and handle common issues.

## What This Website Does

SkillHighlight Analyzer helps users review a resume by:

- Extracting text from a resume file.
- Highlighting important resume keywords.
- Showing the balance of hard skills, soft skills, recruiter keywords, and action verbs.
- Scoring each sentence based on how strongly it presents achievements.
- Giving an overall self-promotion score.

The website is meant to support resume improvement. It should not be treated as a final hiring decision tool.

## Supported Inputs

You can use any of these input methods:

- Upload a PDF resume.
- Upload a DOCX resume.
- Upload a TXT resume.
- Paste resume text directly into the text box.

For best results, use a clean DOCX or text-based PDF. Scanned image-only PDFs may not extract correctly.

## How To Open The Website

From the project folder, run:

```bash
streamlit run main_modular.py
```

Streamlit will open the website in your browser. If it does not open automatically, copy the local URL shown in the terminal, usually something like:

```text
http://localhost:8501
```

## Basic Usage

1. Open the website.
2. Upload your resume using the file uploader.
3. If you do not want to upload a file, paste your resume text into the text area.
4. Wait for the analyzer to process the resume.
5. Review the highlighted resume text.
6. Check the keyword category percentages.
7. Review the self-promotion score.
8. Read the sentence-by-sentence feedback.
9. Edit your resume outside the website, then upload or paste the revised version again.

## Main Screen Sections

### Resume Input

At the top of the page, you can either upload a file or paste text.

If both are available, the uploaded file is used first. To analyze pasted text instead, remove the uploaded file or refresh the app and paste the text directly.

### Extracted / Input Text

This section shows the resume text after extraction and cleanup.

Highlighted words show detected resume keywords. The colors represent different keyword categories.

### Keyword Analysis

This section shows keyword category percentages. The percentages describe the balance of detected keyword matches, not a final resume grade.

For example, if Hard Skills shows `40%`, that means 40% of the detected keyword matches belonged to the hard-skill category.

### Self-Promotion Score

This score summarizes how achievement-focused the resume sounds.

General interpretation:

| Score Range | Meaning |
| --- | --- |
| Greater than `0.8` | Strong achievement-focused language |
| Greater than `0.5` up to `0.8` | Moderate to good self-promotion |
| `0.5` or lower | Needs stronger achievement language |

A low score does not mean the resume is bad. It usually means the resume has more responsibility descriptions than measurable results.

### Sentence Analysis

This section lists individual sentences and labels them as:

- Strong
- Moderate
- Weak

Use this section to find sentences that can be rewritten with stronger action verbs, metrics, and outcomes.

## Highlight Colors

| Category | Color | What It Means |
| --- | --- | --- |
| Hard Skills | Teal | Technical skills, tools, programming languages, platforms |
| Soft Skills | Purple | Communication, leadership, teamwork, adaptability |
| Recruiter Keywords | Orange | Resume-friendly or recruiter-facing terms |
| Action Verbs | Red | Verbs that describe action and achievement |

## How To Improve A Resume Using The Results

### If The Self-Promotion Score Is Low

Try adding:

- Numbers or percentages.
- Results and outcomes.
- Stronger action verbs.
- Specific project impact.
- Leadership or ownership language.

Weak example:

```text
Responsible for managing project tasks.
```

Stronger example:

```text
Managed project tasks for a 5-person team and reduced delivery delays by 20%.
```

### If There Are Few Hard Skills

Add relevant tools, programming languages, frameworks, platforms, or technical methods that honestly match your experience.

Examples:

- Python
- SQL
- Excel
- Power BI
- TensorFlow
- Git

### If There Are Few Action Verbs

Rewrite bullet points so they begin with stronger verbs.

Examples:

- Built
- Led
- Improved
- Developed
- Automated
- Managed
- Optimized

### If The Resume Has Too Many Generic Keywords

Try replacing vague phrases with more specific accomplishments.

Instead of:

```text
Good communication skills and hardworking.
```

Use:

```text
Presented weekly project updates to stakeholders and coordinated task handoffs across three teams.
```

## Sidebar Settings

The sidebar contains advanced controls. Most users can leave the defaults unchanged.

### Token-Aligned Mode

Recommended: On.

This makes highlighting safer by matching whole tokens instead of partial word fragments.

### Relax HARD Skills

When enabled, the app becomes more permissive when detecting hard skills.

Use this if the app is missing technical terms that should be highlighted.

### Relax ACTION Verbs

When enabled, the app becomes more permissive when detecting action verbs.

Use this if action verbs are not being highlighted enough.

### Relax SOFT Skills

Default: On.

This allows soft skills to be highlighted more easily, even when sentiment filtering would otherwise suppress them.

### Relax RECRUITER Keywords

Default: On.

This allows recruiter keywords to be detected in shorter resume bullets.

### Soft Skill Sentiment Threshold

This controls when soft-skill matches should be suppressed because the surrounding sentence sounds negative.

Most users do not need to change this.

### Legacy HTML Rendering

This changes the highlight style to a simpler older format.

Use it only if the normal highlighting display has visual issues.

## Common Issues And Fixes

### The App Says To Upload Or Paste Text

The app has not received resume content yet.

Fix:

- Upload a PDF, DOCX, or TXT file.
- Or paste resume text into the text area.

### My PDF Text Looks Messy

Some PDFs have complicated layouts, columns, tables, or image-based text.

Fix:

- Try uploading a DOCX version of the resume.
- Copy and paste the resume text directly.
- Use a simpler resume layout.

### The App Misses Some Keywords

Keyword detection is based on the keyword database and context checks.

Fix:

- Turn on the relevant relaxation setting in the sidebar.
- Check whether the keyword exists in the resume text.
- Use clearer wording or standard skill names.

### Too Many Words Are Highlighted

The matching settings may be too permissive.

Fix:

- Turn off relaxation settings.
- Keep Token-Aligned Mode enabled.

### The Score Seems Too Low

The resume may describe duties more than achievements.

Fix:

- Add measurable results.
- Add outcomes.
- Start bullet points with action verbs.
- Include project scope, tools used, and impact.

### The Website Is Slow The First Time

The app loads NLP and machine learning models when it starts. The first run can take longer than later runs.

Fix:

- Wait for the model loading to finish.
- Keep the app open while testing multiple resumes.

## About The MainThread Warning

You may see this terminal warning:

```text
Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.
```

This is a Streamlit warning. It usually appears when Streamlit code is run outside the normal Streamlit app context, such as running a Streamlit file with plain Python:

```bash
python main_modular.py
```

Recommended fix:

```bash
streamlit run main_modular.py
```

If the website still opens and works, the warning is usually harmless. For demos or normal use, always start the app with `streamlit run`.

## Privacy Reminder

This app processes the resume locally in the running Streamlit session. Users should still avoid uploading resumes with sensitive personal information unless they trust the machine and environment where the app is running.

## Best Practices For Users

- Use a clean resume file.
- Prefer DOCX or text-based PDF files.
- Review the extracted text before trusting the score.
- Treat the score as guidance, not a final judgment.
- Rewrite weak sentences with clearer achievements and measurable impact.
- Re-run the analyzer after editing the resume.

## Quick Start Summary

1. Run `streamlit run main_modular.py`.
2. Upload or paste a resume.
3. Review highlighted keywords.
4. Check keyword category percentages.
5. Review the self-promotion score.
6. Improve weak sentences using action verbs, metrics, and results.
7. Analyze the revised resume again.

