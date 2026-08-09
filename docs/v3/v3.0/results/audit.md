# Pilot integrity audit

Overall: **PASS**

- PASS — dataset validation passed
- PASS — public has 64 unique variants
- PASS — private has 64 matching variants
- PASS — private keys absent from public views
- PASS — all locked call ids present
- PASS — all locked calls succeeded
- PASS — all locked calls have parsed results
- PASS — served model ids match plan
- PASS — scores cover exact locked calls
- PASS — summary has no invalid results
- PASS — estimated paid cost within 15 usd

Locked calls checked: 384/384.
Archived superseded attempts retained: 35.

## Artifact hashes

- `dataset_manifest.json`: `d8d02ef8ed0750cf378cc08eac5a56c59884ceec9330ecce94727142105a692e`
- `verifier_views.jsonl`: `b3ac7b81f608a6c78bc32709759e90d097249fdb0007a2a781af272da9c3aa7f`
- `private_gold.jsonl`: `0135ab6686f46323f7242787f51d3dc3695414c3a346a63c37c0bad8bf41676f`
- `responses.jsonl`: `92d42a073d788df93176aa38678f055b8f0281686a85114a0dbc0a210250aa81`
- `scores.jsonl`: `89d648dd19bd6de54ac32cfde21bfa1d875ddffcd8f8009178cb282826644fc2`
- `summary.json`: `aa557ba78fc95ae316d54ac7ced450ba19c4bb401d1b41a92cc6537966507f26`
