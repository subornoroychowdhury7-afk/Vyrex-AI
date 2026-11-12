# Data governance and quality policy

Before any corpus is downloaded or used, record the source URL/provider, collection date, license/terms, permitted uses, language, domain, approximate size, and processing notes in a source manifest.

Reject or quarantine data when rights are unclear, content is mostly non-English, the material contains credentials/secrets, personal data cannot be handled appropriately, or the source is spam/low quality. Deduplicate exact and near-duplicate documents before splitting. Split by document/source where possible to reduce leakage. Keep test data isolated.

Initial domain tags: `general`, `education`, and `coding`. Report token counts and quality statistics by tag. The proposed 60/25/15 mixture is a sampling target to test, not an instruction to oversample poor-quality material. Licensing and privacy review takes precedence over quantity.
