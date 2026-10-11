<!-- Copyright (c) 2026 Robotics and AI Institute LLC dba RAI Institute. All rights reserved. -->

# Contributing

## What to expect

This is research code released so the results in our paper can be reproduced.
It gets best-effort maintenance only. Replies may be slow, and new features are
unlikely to be taken on.

## Reporting a bug

Open an issue with:

- what you ran and what happened;
- your setup — `./scripts/verify.sh` prints everything relevant;
- whether you are using the released models or your own.

Check the README's Limitations section and `docs/GRANDTOUR.md` first. Most
problems come from a TensorRT version mismatch, missing model files, or wrong
topic and frame names.

## Sending a fix

Bug fixes are welcome. For anything larger, open an issue first — we may not be
able to maintain it.

Before opening a pull request:

- keep the change focused and match the surrounding style;
- check it still builds (`colcon build`) and that the demo in
  `docs/GRANDTOUR.md` still runs;
- run `pre-commit run --all-files`;
- do not add GPL, LGPL or AGPL dependencies, and say what licence anything new
  is under;
- if you copy code from another project, keep its attribution and add it to
  `NOTICE`;
- do not commit models, datasets or recordings. Models are published on the
  Hugging Face Hub.

By sending a contribution you confirm you have the right to release it under
this repository's licence, the RAI Institute Research License (see LICENSE).
