Use the following prompt with Deep Research after attaching these files:

- `latex/main.pdf`
- `deep-research-report.md`
- `paper_analysis.md`
- `work_division.md`
- `dev_work_division.md`
- `walkthrough.md`
- `Uncertainty-Calibrated Collision-Risk Forecasting in Autonomous Driving Using Spatiotemporal Interac.pdf`

Prompt:

```text
You are doing deep research and document generation for a new project proposal.

You have 2 kinds of attached inputs:

1. `main.pdf`
This is the target project you must work on.

2. A solved reference package from a different project:
- `Uncertainty-Calibrated Collision-Risk Forecasting in Autonomous Driving Using Spatiotemporal Interac.pdf`
- `deep-research-report.md`
- `paper_analysis.md`
- `work_division.md`
- `dev_work_division.md`
- `walkthrough.md`

Your job is to reverse-engineer how the reference project was transformed from a short proposal PDF into a full execution/research package, then produce the same caliber of package for `main.pdf`.

Core rule:
- Treat the reference package as a template for deliverable type, structure, level of detail, and planning rigor.
- Do not copy domain content from the reference project.
- All project-specific content must be derived from `main.pdf` plus current external research.
- If the reference package and `main.pdf` suggest different assumptions, `main.pdf` wins.

Operating mode:
- Think like a research lead, senior ML systems engineer, and technical program manager at the same time.
- This is not a generic summary task. The output must be implementation-first, execution-oriented, and immediately useful for building the project.
- Use current web research to fill in missing technical details, baselines, datasets, model choices, tools, open-source repos, evaluation protocols, and deployment options.
- Prefer primary sources: official papers, project websites, official GitHub repos, benchmark pages, docs.
- Verify claims that are likely to change over time, especially repo/code availability, benchmark status, model releases, and tooling.
- Keep the project scoped to a realistic course-project / semester-project budget unless `main.pdf` clearly demands more.
- Do not ask clarifying questions. Make reasonable assumptions, state them explicitly, and keep moving.

What to infer from `main.pdf`:
- Exact project title
- Exact author/team-member names
- Core problem statement
- Supported tasks
- Inputs and outputs
- System modules
- Evaluation goals
- Feasibility constraints
- Expected final deliverables

What to infer from the reference package:
- What “good” looks like for depth, organization, and rigor
- How much detail belongs in the deep research report versus the paper analysis
- How the work division and personal implementation plan should be structured
- How the walkthrough summarizes the artifacts and validates the work

Deliverables to produce

Produce all of the following files for the `main.pdf` project:

1. `deep-research-report.md`
Purpose:
- A master implementation-first blueprint for the project.

Required characteristics:
- Much deeper than the original proposal
- Not just a literature survey
- Rich with build choices, architectural options, tradeoffs, benchmarks, datasets, risks, and execution guidance
- Tailored to the actual project in `main.pdf`

Required sections:
- Title, authors, date
- Scope, success criteria, and precise problem definition
- System architecture blueprints
- Model / method roster and current buildable landscape
- Dataset options and preprocessing strategy
- Training pipeline and experiment plan
- Evaluation framework and metrics
- Runtime / deployment / export plan
- Failure modes and mitigation strategies
- Reproducibility checklist
- Week-by-week execution plan
- Prioritized action items
- Open bibliography / code links

Project-specific adaptation requirement:
- If `main.pdf` is about a language-guided vision model factory, then this report must deeply cover language parsing, constraint extraction, label resolution, recipe/model selection, constrained training, export packaging, and support for the actual scoped tasks in the proposal.
- It must also identify realistic model families and baselines for the supported tasks, compare feasible backbone choices, and define how natural-language specifications are converted into concrete training configurations.

2. `paper_analysis.md`
Purpose:
- A structured, readable analysis of the project and its design space.

Required characteristics:
- More compact than the deep research report
- Still technically serious
- Good for a teammate who wants to understand the project quickly but thoroughly

Required sections:
- Executive summary
- Problem formulation and scope
- Architecture deep-dive
- Model/method landscape
- Dataset and preprocessing analysis
- Evaluation framework
- Training/deployment strategy
- Schedule
- Risks / failure modes
- Strengths
- Potential gaps / concerns

3. `work_division.md`
Purpose:
- A complete team-wide execution plan.

Required characteristics:
- Use the actual author names from `main.pdf`
- If the team has 3 members, mirror the reference package’s level of specificity
- If the team size differs, adapt the structure responsibly

Required sections:
- Ownership domains
- One section per team member with:
  - domain
  - week-by-week deliverables
  - key files / components owned
- Shared responsibilities
- RACI-style ownership matrix
- Weekly sync structure
- Critical integration / handoff points
- Decision checkpoints

Project-specific adaptation requirement:
- The ownership split must reflect the actual system modules in `main.pdf`.
- For a language-guided vision project, a natural split would likely include:
  - language/configuration core
  - vision/data/training/evaluation
  - systems/integration/export/deployment
- But do not force this if a better split is implied by the proposal.

4. `dev_work_division.md`
Purpose:
- A personal master implementation plan for Dev Sanghvi if Dev is one of the authors.
- If Dev Sanghvi is not on the author list, create the same style of document for the most technically central contributor and say so clearly in the document.

Required characteristics:
- Very concrete
- Phase-based
- Week-level deliverables
- Includes validation expectations and tuning knobs

Required sections:
- Phase breakdown
- Weekly deliverables
- Concrete subcomponents to implement
- Validation performed / acceptance checks
- Key magic numbers / hyperparameters / decision knobs to tune
- Final summary checklist

5. `walkthrough.md`
Purpose:
- A short meta-document that explains what was produced and why.

Required sections:
- What was done
- Artifacts created
- What each artifact covers
- Key findings
- Validation

Cross-file consistency requirements

All files must agree on:
- Project title
- Team-member names
- Scope
- Task definitions
- Core architecture
- Chosen baselines
- Dataset story
- Timeline
- Risks

Quality bar

- Make the package feel like it was produced by someone preparing to actually build the system, not just write about it.
- Be concrete about implementation paths, not vague.
- Name realistic tools, libraries, model families, repos, and benchmarks.
- Distinguish clearly between MVP choices and stretch goals.
- Call out what is feasible under modest compute and time budgets.
- Surface unresolved decisions explicitly.
- When multiple viable paths exist, recommend one default path and briefly justify it.
- Use clean markdown.
- Do not leave raw tool artifacts or malformed citation tokens such as `cite`, `filecite`, `turn0...`, or similar strings in the final files.

Output format

Return the answer as 5 file blocks in this exact order:

=== BEGIN FILE: deep-research-report.md ===
...content...
=== END FILE: deep-research-report.md ===

=== BEGIN FILE: paper_analysis.md ===
...content...
=== END FILE: paper_analysis.md ===

=== BEGIN FILE: work_division.md ===
...content...
=== END FILE: work_division.md ===

=== BEGIN FILE: dev_work_division.md ===
...content...
=== END FILE: dev_work_division.md ===

=== BEGIN FILE: walkthrough.md ===
...content...
=== END FILE: walkthrough.md ===

Do not add any extra commentary outside those file blocks.
```
