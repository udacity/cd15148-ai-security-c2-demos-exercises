# Purpose of This Repo

This repo is the source of truth for all exercises in this course.

> IMPORTANT!  Please remove these instructions before sharing this repo with learners.

## Folder Structure

This repo contains one folder for each Implementation (Apply) module in the course. Each module folder holds a `demo/` and an `exercise/` subdirectory, and each `exercise/` is split into `starter/` and `solution/`:

```bash
module-#-<module-slug>/
├── demo/
│   └── .gitkeep
└── exercise/
    ├── starter/
    │   └── INSTRUCTIONS.md
    └── solution/
        └── .gitkeep
```

- `demo/` - Contains the instructor-led demo materials for the module.
- `exercise/starter/` - Contains the starter files and instructions for the exercise (INSTRUCTIONS.md template provided).
- `exercise/solution/` - Contains the solution files for the exercise.

> **Note:** The `.gitkeep` files preserve empty directory structure in the repository. Remove a `.gitkeep` once real content is added to its folder.

Only Implementation (Apply) modules live in this repo — Conceptual (Understand) modules are slides-based and have no demo or exercise.

**The `module-#` folder prefixes are historical and no longer match the classroom.** They record each module's position in the original 22-module sequence, where every Apply module followed its paired Understand module, which is why they are all odd. Two modules were later dropped and the course now ships 17, so the classroom numbers have shifted. The folder names were deliberately left as they are, because classroom pages link to these paths and renaming them would break those links. Read the folder number as an identifier, not a position.

The 8 Implementation modules in this course are:

| Folder | Module Title | Classroom module |
| ------ | ------------ | ---------------- |
| `module-3-apply-ai-red-teaming` | Apply AI Red Teaming | M3 |
| `module-7-apply-evasion-attacks` | Apply Evasion Attacks | M5 |
| `module-9-apply-data-poisoning` | Apply Data Poisoning | M7 |
| `module-11-apply-prompt-injection` | Apply Prompt Injection | M9 |
| `module-13-apply-vector-database-attacks` | Apply Vector Database Attacks | M11 |
| `module-15-apply-model-inversion` | Apply Model Inversion | M13 |
| `module-19-apply-quantitative-robustness-testing` | Apply Quantitative Robustness Testing | M15 |
| `module-21-apply-ai-supply-chain-vulnerability-scanning` | Apply AI Supply Chain Vulnerability Scanning | M17 |

Two modules that once lived here were dropped from the course and are not part of the build:

- **Apply LLM-Assisted Vulnerability Discovery** — the `module-5-apply-llm-assisted-vulnerability-discovery/` folder is still in the tree but is unused, and nothing in the classroom links to it.
- **Apply AI Red Teaming with Microsoft Counterfit** — the folder was removed, because Counterfit cannot be installed on Python 3.12 or Apple Silicon.

> ⚠️ **DO NOT NUMBER the exercises!**
> Module folders carry a number, for the historical reason above, but the exercise folders inside (`exercise/starter/`, `exercise/solution/`) do not — our modular content may be used in more than one program where the order and number of exercises may differ from the order and number in the primary build.

## Resources for Building Exercises

The [Exercise Creation Resources](Exercise%20Creation%20Resources/) folder contains essential guidelines and standards for creating high-quality, accessible, and engaging exercises. These resources ensure consistency and help you follow best practices when developing course content.

### [Exercise Guidance.md](Exercise%20Creation%20Resources/Exercise%20Guidance.md)

Comprehensive guide covering exercise design principles, instruction writing, starter and solution code best practices, and requirements for solution videos and text. This is your primary resource for understanding what makes an effective exercise.

### [Accessibility Standards.md](Exercise%20Creation%20Resources/Accessibility%20Standards.md)

Details the WCAG 2.1 AA accessibility standards that all content must meet, including guidelines for headings, alt text, hyperlinks, color contrast, and avoiding images of text. Ensures exercises are accessible to all learners regardless of their abilities or use of assistive technology.

### [Real-World Content Guidelines.md](Exercise%20Creation%20Resources/Real-World%20Content%20Guidelines.md)

Guidelines for using real-world examples, company logos, trademarks, and references to people and organizations in exercises. Covers when it's appropriate to use actual brands versus creating fictitious examples and how to avoid legal and ethical issues.

### [Third Party Images and Datasets.md](Exercise%20Creation%20Resources/Third%20Party%20Images%20and%20Datasets.md)

Requirements for using third-party content including licensing requirements (Creative Commons, public domain), attribution standards, and approved sources for images, coding libraries, and datasets. Lists acceptable and unacceptable license types for commercial educational use.
