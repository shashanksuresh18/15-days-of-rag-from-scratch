# Sample corpus: Acme handbook (synthetic)

`acme-handbook/` contains eight short internal policy documents for **Acme Corp, a fictional company**. They were written from scratch for this repository so that every learner can run the examples without downloading anything, paying for anything or worrying about reuse rights.

| File | Owner | Last reviewed |
|---|---|---|
| `remote-work-policy.md` | People Team | 2026-03-02 |
| `expense-reimbursement-policy.md` | Finance | 2026-01-15 |
| `password-and-mfa-standard.md` | IT Security | 2026-05-20 |
| `vpn-access-guide.md` | IT Service Desk | 2026-04-08 |
| `parental-leave-policy.md` | People Team | 2025-11-10 |
| `security-incident-reporting.md` | Security Team | 2026-06-30 |
| `data-retention-policy.md` | Legal | 2026-02-12 |
| `new-starter-it-checklist.md` | IT Service Desk | 2026-07-01 |

## Why a fictional company?

A language model cannot have seen Acme's policies during training, because they did not exist. That makes the corpus a clean test of RAG: the only way to answer "How much is the home office allowance?" correctly is to retrieve the right passage.

## Format

Each file starts with a small front-matter block (`title`, `owner`, `last_reviewed`) followed by `##` sections. Day 01 splits documents on those `##` headings. Later days revisit parsing, metadata and chunking in more depth.

## Rules

- **Not real policy advice.** Numbers such as "USD 500" or "16 weeks" are invented for teaching.
- **No personal data.** Email addresses and URLs use the reserved `.example` domain (RFC 2606) and cannot reach anyone.
- **Licence:** covered by the repository licence (see the root `README.md`).
- **Changing the corpus changes results.** The notebooks contain sanity checks that expect these exact files. If you edit them, rerun the notebook and update the checks.
