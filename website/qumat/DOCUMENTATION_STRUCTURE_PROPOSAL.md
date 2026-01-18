---
layout: page
title: Documentation Structure Proposal
---

# Documentation Structure Proposal

This document proposes an improved organization for Mahout Qumat documentation to make it more maintainable and user-friendly.

## Current Issues

1. **Content Duplication**: Documentation exists in multiple places:
   - `/website/qumat/` - Website documentation (was empty, now filled)
   - `/qdp/DEVELOPMENT.md` - QDP development guide
   - `/qdp/qdp-python/README.md` - Python package documentation
   - `/README.md` - Main project README
   - `/docs/` - Additional documentation

2. **Inconsistent Structure**: Different documentation areas use different formats and organization

3. **Discoverability**: Hard to find specific information due to scattered files

4. **Maintenance**: Updates require changes in multiple locations

## Proposed Structure

### Option A: Centralized Website Documentation (Recommended)

Consolidate all user-facing documentation under `/website/qumat/`:

```
website/qumat/
├── index.md                              # Main Qumat landing page
├── DOCUMENTATION_STRUCTURE_PROPOSAL.md   # This file
│
├── core/                                 # Qumat Core documentation
│   ├── index.md                          # Core overview
│   ├── getting-started/
│   │   └── index.md                      # Installation & quick start
│   ├── concepts/
│   │   └── index.md                      # Key concepts
│   ├── api/
│   │   └── index.md                      # API reference
│   ├── examples/
│   │   └── index.md                      # Code examples
│   └── guides/                           # NEW: Advanced guides
│       ├── backends.md                   # Backend-specific info
│       ├── best-practices.md             # Best practices
│       └── troubleshooting.md            # Common issues
│
├── qdp/                                  # QDP documentation
│   ├── index.md                          # QDP overview
│   ├── getting-started/
│   │   └── index.md                      # Installation & quick start
│   ├── concepts/
│   │   └── index.md                      # Key concepts
│   ├── api/
│   │   └── index.md                      # API reference
│   ├── examples/
│   │   └── index.md                      # Code examples
│   ├── guides/                           # NEW: Advanced guides
│   │   ├── file-formats.md               # File format support
│   │   ├── performance.md                # Performance tuning
│   │   └── multi-gpu.md                  # Multi-GPU setup
│   └── development/                      # NEW: Development docs
│       ├── index.md                      # Development overview
│       ├── building.md                   # Build instructions
│       ├── testing.md                    # Testing guide
│       └── contributing.md               # Contribution guide
│
├── quantum-computing-primer/             # Educational content
│   └── ... (existing structure)
│
└── papers/                               # Research papers
    └── ... (existing structure)
```

### Supporting Files in Code Directories

Keep minimal README files in code directories with links to main documentation:

```
qdp/
├── README.md                 # Brief overview + link to website/qumat/qdp/
├── DEVELOPMENT.md            # MOVE to website/qumat/qdp/development/
└── qdp-python/
    └── README.md             # Brief overview + link to website/qumat/qdp/

qumat/
└── README.md                 # NEW: Brief overview + link to website/qumat/core/

examples/
└── README.md                 # NEW: Index of examples + link to website/qumat/
```

## Migration Plan

### Phase 1: Immediate (Completed ✓)
- [x] Fill empty documentation pages in `/website/qumat/`
- [x] Add comprehensive content to all sections

### Phase 2: Short-term (Recommended)
- [ ] Create `guides/` subdirectories for advanced topics
- [ ] Move `/qdp/DEVELOPMENT.md` to `/website/qumat/qdp/development/`
- [ ] Create brief README files in code directories linking to website docs
- [ ] Add cross-references between related documentation pages
- [ ] Create `/examples/README.md` as an index

### Phase 3: Medium-term (Optional)
- [ ] Set up automated documentation generation from docstrings (Sphinx/pdoc)
- [ ] Add search functionality to website
- [ ] Create interactive tutorials/notebooks
- [ ] Add versioned documentation for different releases

## Benefits of This Structure

1. **Single Source of Truth**: Website is the primary documentation location
2. **Consistent Navigation**: Parallel structure for Core and QDP
3. **Better Discoverability**: Clear hierarchy makes finding information easier
4. **Easier Maintenance**: Updates in one place rather than multiple files
5. **User-Friendly**: Progressive disclosure from getting-started → concepts → guides → API
6. **Developer-Friendly**: Separate development docs from user docs

## Alternative Structures Considered

### Option B: Documentation in Code Directories

Keep docs near code (e.g., `/qumat/docs/`, `/qdp/docs/`) and generate website from them.

**Pros:**
- Docs closer to code
- Easier for developers to update

**Cons:**
- Harder to maintain consistent structure
- Website requires build step
- Less accessible to non-developers

**Decision:** Not recommended due to maintenance overhead

### Option C: Docs Repository

Separate repository for documentation.

**Pros:**
- Clean separation
- Easier to manage large doc sets

**Cons:**
- Adds complexity
- Harder to keep in sync with code
- Overkill for current project size

**Decision:** Not recommended for current scale

## Implementation Guidelines

1. **Write for the Audience**:
   - Getting Started: New users, step-by-step
   - Concepts: Intermediate users, understanding
   - Guides: Advanced users, specific tasks
   - API: Reference, completeness

2. **Use Consistent Formatting**:
   - Code blocks with syntax highlighting
   - Clear headings and sections
   - Examples for each concept
   - Links to related topics

3. **Keep It DRY (Don't Repeat Yourself)**:
   - Use links instead of duplicating content
   - Maintain single source for each piece of information
   - Use includes for shared content (if Jekyll supports)

4. **Make It Navigable**:
   - Breadcrumbs showing current location
   - "Next Steps" sections linking to related pages
   - Clear table of contents for long pages

5. **Test the User Journey**:
   - Can a new user get started in < 5 minutes?
   - Can they find API details easily?
   - Are advanced topics discoverable?

## Success Metrics

- Reduction in "where is X documented?" questions
- Faster onboarding for new contributors
- Fewer documentation-related issues
- Positive user feedback on documentation

## Next Steps

1. Review this proposal with the team
2. Decide on Option A vs alternatives
3. Create issues for Phase 2 tasks
4. Assign owners for migration tasks
5. Set timeline for completion

## Questions and Feedback

Please provide feedback via:
- GitHub Issues
- Mailing list discussion
- PR comments on this proposal

---

Last Updated: 2026-01-18
Authors: GitHub Copilot
Status: Proposed
