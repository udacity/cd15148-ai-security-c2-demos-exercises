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

Only Implementation (Apply) modules live in this repo — Conceptual (Understand) modules are slides-based and have no demo or exercise. The `module-#` prefix on each folder reflects the module's position in the full course sequence (which is why the numbering skips even positions: those are the paired Understand modules).

The 10 Implementation modules in this course are:

| # | Folder | Module Title |
| -- | ------ | ------------ |
| 3 | `module-3-apply-ai-red-teaming` | Apply AI Red Teaming |
| 5 | `module-5-apply-llm-assisted-vulnerability-discovery` | Apply LLM-Assisted Vulnerability Discovery |
| 7 | `module-7-apply-evasion-attacks` | Apply Evasion Attacks |
| 9 | `module-9-apply-data-poisoning` | Apply Data Poisoning |
| 11 | `module-11-apply-prompt-injection` | Apply Prompt Injection |
| 13 | `module-13-apply-vector-database-attacks` | Apply Vector Database Attacks |
| 15 | `module-15-apply-model-inversion` | Apply Model Inversion |
| 17 | `module-17-apply-ai-red-teaming-with-microsoft-counterfit` | Apply AI Red Teaming with Microsoft Counterfit |
| 19 | `module-19-apply-quantitative-robustness-testing` | Apply Quantitative Robustness Testing |
| 21 | `module-21-apply-ai-supply-chain-vulnerability-scanning` | Apply AI Supply Chain Vulnerability Scanning |

> ⚠️ **DO NOT NUMBER the exercises!**
> Module folders are numbered to mirror course position, but the exercise folders inside (`exercise/starter/`, `exercise/solution/`) are not — our modular content may be used in more than one program where the order and number of exercises may differ from the order and number in the primary build.

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
