"""Step 5, part 2: run llama.cpp's converter (portable/llama.cpp/convert/) with
the bundled Python. Arguments go straight through to convert_hf_to_gguf.py.

    python Step_5/run_convert.py <model folder> --outfile x.gguf --outtype f16

Two things the converter assumes that the bundle does not have:
  - its own folder on sys.path (the bundled Windows Python leaves it off);
  - sentencepiece installed. Llama 3 does not use it, but the converter tries
    the sentencepiece tokenizer first and only moves on to the Llama 3 one
    when that attempt raises FileNotFoundError (no tokenizer.model). Without
    the package the attempt raises ImportError instead, which it does not
    catch. So a missing package is reported as FileNotFoundError, the same
    as a missing tokenizer.model. (Not a stand-in module: transformers checks
    for the package on import and fails on a fake one.)
"""
import os
import runpy
import sys

CONVERT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "portable", "llama.cpp", "convert")
SCRIPT = os.path.join(CONVERT, "convert_hf_to_gguf.py")


def skip_sentencepiece_if_missing():
    import conversion.base as base
    original = base.TextModel._create_vocab_sentencepiece

    def create_vocab_sentencepiece(self):
        try:
            import sentencepiece  # noqa: F401
        except ImportError:
            raise FileNotFoundError("sentencepiece is not installed: not a sentencepiece model")
        return original(self)
    base.TextModel._create_vocab_sentencepiece = create_vocab_sentencepiece


def main():
    sys.path.insert(0, CONVERT)
    skip_sentencepiece_if_missing()
    sys.argv = [SCRIPT] + sys.argv[1:]
    runpy.run_path(SCRIPT, run_name="__main__")


if __name__ == "__main__":
    main()
