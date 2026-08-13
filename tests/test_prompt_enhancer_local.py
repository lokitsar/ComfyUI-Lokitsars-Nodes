import importlib
import shutil
import sys
import tempfile
import types
import unittest
import uuid
from pathlib import Path


class _FakeRoutes:
    def get(self, _path):
        return lambda fn: fn

    def post(self, _path):
        return lambda fn: fn


def _load_nodes_module():
    output_dir = Path(tempfile.gettempdir()) / f"prompt-enhancer-{uuid.uuid4().hex}"
    output_dir.mkdir(parents=True)
    fake_aiohttp = types.ModuleType("aiohttp")
    fake_aiohttp.web = types.SimpleNamespace(
        Response=lambda *args, **kwargs: None,
        FileResponse=lambda *args, **kwargs: None,
        json_response=lambda payload, **kwargs: payload,
    )
    sys.modules["aiohttp"] = fake_aiohttp

    fake_server = types.ModuleType("server")
    fake_server.PromptServer = type(
        "PromptServer",
        (),
        {"instance": types.SimpleNamespace(routes=_FakeRoutes())},
    )
    sys.modules["server"] = fake_server

    fake_folder_paths = types.ModuleType("folder_paths")
    fake_folder_paths.get_output_directory = lambda: str(output_dir)
    fake_folder_paths.get_filename_list = lambda _kind: ["qwen-test.safetensors"]
    sys.modules["folder_paths"] = fake_folder_paths

    sys.modules.pop("nodes", None)
    return importlib.import_module("nodes"), output_dir


class _FakeClip:
    def __init__(self):
        self.prompt = None
        self.tokenize_kwargs = None
        self.generate_kwargs = None

    def tokenize(self, prompt, **kwargs):
        self.prompt = prompt
        self.tokenize_kwargs = kwargs
        return {"tokens": [1, 2]}

    def generate(self, _tokens, **kwargs):
        self.generate_kwargs = kwargs
        return [3, 4]

    def decode(self, _tokens):
        return "A clean enhanced prompt."


class TestPromptEnhancerLocal(unittest.TestCase):
    def test_connected_clip_generates_without_http_backend(self):
        nodes, output_dir = _load_nodes_module()
        self.addCleanup(shutil.rmtree, output_dir, True)
        clip = _FakeClip()

        result = nodes.PromptEnhancer().enhance(
            backend="ComfyUI Local",
            prompt="rough idea",
            system_prompt="Expand the idea.",
            max_tokens=222,
            seed=17,
            clip=clip,
        )

        self.assertEqual(result, ("A clean enhanced prompt.",))
        self.assertIn("Expand the idea.", clip.prompt)
        self.assertIn("rough idea", clip.prompt)
        self.assertEqual(clip.generate_kwargs["max_length"], 222)
        self.assertEqual(clip.generate_kwargs["seed"], 17)

    def test_local_backend_requires_selection_or_connection(self):
        nodes, output_dir = _load_nodes_module()
        self.addCleanup(shutil.rmtree, output_dir, True)

        with self.assertRaisesRegex(RuntimeError, "choose a local_text_encoder"):
            nodes.PromptEnhancer().enhance(
                backend="ComfyUI Local",
                prompt="rough idea",
                system_prompt="Expand the idea.",
            )


if __name__ == "__main__":
    unittest.main()
