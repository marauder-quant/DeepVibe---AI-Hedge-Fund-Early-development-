# Trading Bot Tasks Directory

This directory contains individual task files for the trading bot project. Each file represents a specific development task, enhancement, or bug fix that needs to be implemented.

## Directory Structure

Each task is organized in its own markdown file with a standardized format:

```
tasks/
├── README.md                         # This file
├── task_04_after_hours_fix.md        # After hours trading fixes
├── task_05_china_scanner.md          # China scanner and allocation
├── task_26_fix_qmac_oos_results.md   # Fix QMAC out-of-sample results
├── task_27_fix_limit_price_error.md  # Fix sub-penny limit price error
└── ... (other task files)
```

## Task File Format

Each task file follows a consistent structure:

1. **Description** - Brief overview of the task
2. **Background** - Context and information about why this task is needed
3. **Objectives** - Specific goals to accomplish with this task
4. **Instructions for AI Agent** - Step-by-step guidance for completing the task
5. **Success Criteria** - Measurable outcomes that indicate task completion
6. **Resources** - References and materials that may help with the task

## How to Use These Task Files

### For AI Agents

1. Select a task file to work on
2. Review the entire file to understand the context and objectives
3. Follow the step-by-step instructions in the "Instructions for AI Agent" section
4. Utilize the trading bot codebase to implement the necessary changes
5. Test your implementation against the success criteria
6. Document your changes and results

### For Human Developers

1. Use these task files as structured specifications for development work
2. Assign tasks to team members based on expertise and priority
3. Track progress using the outlined objectives and success criteria
4. Reference related tasks when working on interdependent components

## Priority Tasks

The following tasks are currently considered high priority:

1. Task #26: Fix QMAC out-of-sample results
2. Task #27: Fix sub-penny limit price error
3. Task #4: After hours fix, limit orders, and fill execution
4. Task #5: Working China scanner and China allocation

## Task Status Tracking

When working on a task, update its status by adding one of the following labels at the top of the file:

- `STATUS: TODO` - Task has not been started
- `STATUS: IN PROGRESS` - Task is currently being worked on
- `STATUS: REVIEW` - Task implementation is complete and awaiting review
- `STATUS: DONE` - Task has been completed and approved

## Contributing New Tasks

To add a new task, create a markdown file following the established format:

```markdown
# Task [Number]: [Task Name]

## Description
[Brief description of the task]

## Background
[Context and reason for the task]

## Objectives
1. [Objective 1]
2. [Objective 2]
...

## Instructions for AI Agent
[Detailed step-by-step instructions]

## Success Criteria
[Measurable outcomes]

## Resources
[Helpful references and materials]
``` 