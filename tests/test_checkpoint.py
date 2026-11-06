"""Offline checkpoint round trips using a tiny randomly initialized encoder."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(
    importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"),
    "Neural checkpoint tests require torch and transformers",
)
class CheckpointTests(unittest.TestCase):
    def test_all_model_types_restore_weights_and_metadata(self):
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

        from src.inference.checkpoint import load_classifier, load_model, load_proposal
        from src.models.mh_bio_crf import XLMRMultiHeadBIOCRF
        from src.models.two_stage.classifier import ZenSpanTechniqueClassifier
        from src.models.two_stage.proposal import SpanProposalDetector

        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1, "hello": 2}, unk_token="[UNK]")),
            unk_token="[UNK]", pad_token="[PAD]",
        )
        for model_type, architecture in (("mhbio_crf", "independent"), ("ts_proposal", "independent"), ("ts_classifier", "shared"), ("ts_classifier", "independent")):
            with self.subTest(model_type=model_type, architecture=architecture), tempfile.TemporaryDirectory() as temp:
                checkpoint = Path(temp)
                encoder = BertModel(BertConfig(vocab_size=3, hidden_size=8, num_hidden_layers=1, num_attention_heads=2, intermediate_size=16))
                if model_type == "mhbio_crf":
                    model = XLMRMultiHeadBIOCRF(encoder, num_labels=2)
                    loader = load_model
                elif model_type == "ts_proposal":
                    model = SpanProposalDetector(encoder)
                    loader = load_proposal
                else:
                    model = ZenSpanTechniqueClassifier(encoder, num_labels=2, hidden_size=5, head_architecture=architecture)
                    loader = load_classifier
                meta = dict(model_type=model_type, labels=["A", "B"], classifier_hidden_size=5, head_architecture=architecture)
                (checkpoint / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
                encoder.save_pretrained(checkpoint / "encoder")
                tokenizer.save_pretrained(checkpoint / "tokenizer")
                torch.save(model.state_dict(), checkpoint / "model.pt")
                restored, restored_tokenizer, restored_meta = loader(checkpoint, torch.device("cpu"))
                self.assertEqual(restored_meta, meta)
                self.assertFalse(restored.training)
                self.assertEqual(restored_tokenizer.get_vocab(), tokenizer.get_vocab())
                self.assertEqual(set(restored.state_dict()), set(model.state_dict()))
                for key, value in model.state_dict().items():
                    self.assertTrue(torch.equal(restored.state_dict()[key], value), key)
                wrong_loader = load_classifier if model_type != "ts_classifier" else load_proposal
                with self.assertRaisesRegex(ValueError, "Expected"):
                    wrong_loader(checkpoint, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
