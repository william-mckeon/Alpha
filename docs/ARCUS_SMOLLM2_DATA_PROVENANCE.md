# Data provenance and recipe adaptation

The executable source specification pins immutable HF revisions. The public small
SmolLM2 recipe does not publish exact component weights or reproduce its educational
DCLM filtering pipeline. Therefore this pipeline is explicitly an **Arcus adaptation**,
not an exact reproduction of the 135M model's training corpus.

Initial document-sampling mix: DCLM baseline 60%, Cosmopedia-v2 15%, FineMath-4plus
10%, InfiWebMath-4plus 5%, and Python Stack-Edu 10%. These are implementation choices,
not reported upstream proportions. Public DCLM baseline is not the filtered variant.
Only Python is selected for the initial code component. Measured token shares and
repeated exposure must accompany results. The specification can be revised before
a campaign; a run's mixture cannot change on resume.

Sources and license references:

- [DCLM baseline](https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0): card lists CC-BY-4.0; retain source provenance.
- [Cosmopedia-v2 corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus): corpus card lists ODC-BY; retain component attribution.
- [FineMath/InfiWebMath](https://huggingface.co/datasets/HuggingFaceTB/finemath): ODC-BY metadata; retain web-source records.
- [Stack-Edu](https://huggingface.co/datasets/HuggingFaceTB/stack-edu): identifiers and per-file terms inherited from The Stack v2; no blanket Apache relicensing. Content retrieval may require Software Heritage access. Keep licenses, repository/path and content hashes when provided.
- [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk), pinned at `5feaf2fd3ffca7c237fc38d1861bc30365d48ffa`: select `apigen-80k` for a later tool SFT stage. It combines Synth-APIGen and xLAM records; respect each source's terms. The top-level card does not provide one license covering all components.
- [Smol-smoltalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk): conversational small-model instruction source; it excludes function calling and cannot fulfill the tool-data requirement alone.

The preparation receipt records availability and omissions. Downloading a bounded
sample does not establish complete corpus coverage, licensing of every document,
or absence of benchmark contamination. Full-corpus and later SFT preparation must
retain these checks; no bulk multi-terabyte download or SFT training is automatic.
